use anyhow::{Context as _, Result};
use collections::HashMap;
use edit_prediction_context::{EditPredictionExcerpt, Line};
use gpui::{App, AppContext, Entity, Task};
use http_client::{Method, Request};
use indoc::indoc;
use language::{Anchor, Buffer, BufferSnapshot, OffsetRangeExt as _, Point};
use open_ai::{FunctionDefinition, MessageContent};
use project::Project;
use serde::Serialize;
use smol::io::AsyncReadExt;
use std::{
    collections::VecDeque,
    fmt::Write,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock},
};

use cloud_zeta2_prompt::{
    Excerpt,
    retrieval_prompt::{SearchToolInput, SearchToolQuery},
    write_codeblock,
};

use crate::Event;

pub static TOOL_SCHEMA: LazyLock<(serde_json::Value, String)> = LazyLock::new(|| {
    let schema = language_model::tool_schema::root_schema_for::<SearchToolInput>(
        language_model::LanguageModelToolSchemaFormat::JsonSchemaSubset,
    );

    let description = schema
        .get("description")
        .and_then(|description| description.as_str())
        .unwrap()
        .to_string();

    (schema.into(), description)
});

pub struct JumpLocation {
    pub buffer: Entity<Buffer>,
    pub anchor: Anchor,
}

#[derive(Serialize)]
struct OpenRouterWrapper {
    #[serde(flatten)]
    request: open_ai::Request,
    provider: OpenRouterProvider,
}

#[derive(Serialize)]
pub struct OpenRouterProvider {
    only: Option<Vec<String>>,
}

pub fn predict_jump(
    active_full_path: Arc<Path>,
    active_buffer: BufferSnapshot,
    cursor_position: Point,
    project: Entity<Project>,
    events: VecDeque<Event>,
    cx: &mut App,
) -> Task<Result<Option<JumpLocation>>> {
    eprintln!("\n\nRequesting jump");
    let search_queries = cx.background_spawn({
        let http_client = cx.http_client().clone();
        async move {
            let prompt =
                build_jump_prompt(&active_full_path, &active_buffer, cursor_position, events)?;

            let (tool_schema, tool_description) = TOOL_SCHEMA.clone();

            let request_body = OpenRouterWrapper {
                request: open_ai::Request {
                    model: "qwen3:8b".into(),
                    // model: "qwen/qwen3-coder-30b-a3b-instruct".into(),
                    messages: vec![open_ai::RequestMessage::User {
                        content: open_ai::MessageContent::Plain(prompt),
                    }],
                    stream: false,
                    max_completion_tokens: None,
                    stop: Default::default(),
                    temperature: 0.7,
                    tool_choice: None,
                    parallel_tool_calls: None,
                    tools: vec![open_ai::ToolDefinition::Function {
                        function: FunctionDefinition {
                            name: cloud_zeta2_prompt::retrieval_prompt::TOOL_NAME.to_string(),
                            description: Some(tool_description),
                            parameters: Some(tool_schema),
                        },
                    }],
                    prompt_cache_key: None,
                    reasoning_effort: None,
                },
                provider: OpenRouterProvider {
                    only: Some(vec!["nebius/fp8".into()]),
                },
            };

            let request = Request::builder()
                .method(Method::POST)
                .uri("http://localhost:11434/v1/chat/completions")
                // .uri("https://openrouter.ai/api/v1/chat/completions")
                // .header(
                //     "Authorization",
                //     format!("Bearer {}", std::env::var("OPENROUTER_API_KEY").unwrap()),
                // )
                // .header("Content-Type", "application/json")
                // .header("HTTP-Referer", "https://zed.dev")
                // .header("X-Title", "Zed Editor")
                .body(serde_json::to_string(&request_body)?.into())?;

            let mut response = http_client.send(request).await?;
            let mut buf = Vec::new();
            response.body_mut().read_to_end(&mut buf).await?;

            if !response.status().is_success() {
                anyhow::bail!("Jump request failed: {}", String::from_utf8_lossy(&buf));
            }

            let response: open_ai::Response = serde_json::from_slice(&buf)?;
            dbg!(&response);

            anyhow::Ok((request_body, response))
        }
    });

    let http_client = cx.http_client().clone();

    cx.spawn(async move |cx| {
        let (mut request_body, mut response) = search_queries.await?;

        let choice = response
            .choices
            .pop()
            .context("No choices in jump response")?;
        let open_ai::RequestMessage::Assistant {
            content: _,
            tool_calls,
        } = &choice.message
        else {
            anyhow::bail!("Jump response didn't include an assistant message");
        };

        let mut queries: Vec<SearchToolQuery> = Vec::new();
        let mut tool_call_id = None;

        for tool_call in tool_calls {
            tool_call_id.get_or_insert(tool_call.id.clone());
            let open_ai::ToolCallContent::Function { function } = &tool_call.content;
            if function.name != cloud_zeta2_prompt::retrieval_prompt::TOOL_NAME {
                log::warn!(
                    "Jump response tried to call an unknown tool: {}",
                    function.name
                );

                continue;
            }

            let input: SearchToolInput = serde_json::from_str(&function.arguments)
                .with_context(|| format!("invalid search json {}", &function.arguments))?;
            queries.extend(input.queries);
        }

        let Some(tool_call_id) = tool_call_id else {
            anyhow::bail!("No searches in jump response");
        };

        if queries.is_empty() {
            anyhow::bail!("No queries in jump response");
        }

        let results =
            zeta2::retrieval_search::run_retrieval_searches(queries, project.clone(), None, cx)
                .await?;

        if results.is_empty() {
            return anyhow::Ok(None);
        }

        // todo! move to background

        let mut combined_results = String::new();
        let mut result_buffers = HashMap::default();

        for (buffer, ranges) in results {
            let (snapshot, full_path) = buffer.read_with(cx, |buffer, cx| {
                (
                    buffer.snapshot(),
                    buffer
                        .file()
                        // todo! use full path but allow matching on just path
                        .map(|file| file.path().as_std_path())
                        .unwrap_or_else(|| Path::new("untitled"))
                        .to_path_buf(),
                )
            })?;

            let ranges = ranges
                .into_iter()
                .map(|range| {
                    let point_range = range.to_point(&snapshot);
                    Line(point_range.start.row)..Line(point_range.end.row)
                })
                .collect::<Vec<_>>();

            let excerpts = zeta2::assemble_excerpts::assemble_excerpts(&snapshot, ranges);
            write_codeblock(
                &full_path,
                &excerpts,
                &[],
                Line(snapshot.max_point().row),
                true,
                &mut combined_results,
            );

            result_buffers.insert(full_path.clone(), (buffer, snapshot));
        }
        eprintln!("{combined_results}");

        request_body.request.tools.clear();
        request_body.request.messages.extend([
            choice.message,
            open_ai::RequestMessage::Tool {
                content: MessageContent::Plain(combined_results),
                tool_call_id,
            },
            open_ai::RequestMessage::User {
                content: MessageContent::Plain(JUMP_INSTRUCTIONS.into()),
            },
        ]);

        let request = Request::builder()
            .method(Method::POST)
            .uri("http://localhost:11434/v1/chat/completions")
            // .uri("https://openrouter.ai/api/v1/chat/completions")
            // .header(
            //     "Authorization",
            //     format!("Bearer {}", std::env::var("OPENROUTER_API_KEY").unwrap()),
            // )
            // .header("Content-Type", "application/json")
            // .header("HTTP-Referer", "https://zed.dev")
            // .header("X-Title", "Zed Editor")
            .body(serde_json::to_string(&request_body)?.into())?;

        let mut response = http_client.send(request).await?;
        let mut buf = Vec::new();
        response.body_mut().read_to_end(&mut buf).await?;
        dbg!(String::from_utf8_lossy(&buf));

        if !response.status().is_success() {
            anyhow::bail!("Jump request failed: {}", String::from_utf8_lossy(&buf));
        }

        let mut response: open_ai::Response = serde_json::from_slice(&buf)?;

        if response.choices.is_empty() {
            return anyhow::Ok(None);
        }

        let choice = response
            .choices
            .pop()
            .context("No choices in jump response")?;

        let open_ai::RequestMessage::Assistant {
            content: Some(MessageContent::Plain(response)),
            tool_calls: _,
        } = &choice.message
        else {
            anyhow::bail!("Jump response didn't include an assistant message");
        };

        dbg!(response);

        let (file_path, line) = response
            .trim()
            .split_once("```jump")
            .context("Missing open fence")?
            .1
            .split_once("```")
            .context("Missing closing fence")?
            .0
            .trim()
            .split_once(":")
            .context("Invalid jump response")?;

        dbg!(file_path, line);

        let line = line.parse::<u32>()?;

        let (buffer, snapshot) = result_buffers
            .get(Path::new(file_path))
            .context("File not found in search results")?;

        anyhow::Ok(Some(JumpLocation {
            buffer: buffer.clone(),
            anchor: snapshot.anchor_after(Point::new(line, 0)),
        }))
    })
}

pub fn build_jump_prompt(
    active_full_path: &Path,
    active_buffer: &BufferSnapshot,
    cursor_position: Point,
    events: VecDeque<Event>,
) -> Result<String> {
    let mut prompt = SEARCH_INSTRUCTIONS.to_string();

    if !events.is_empty() {
        writeln!(&mut prompt, "\n## User Edits\n\n")?;
        write_events(&mut prompt, events);
    }

    writeln!(&mut prompt, "## Cursor context\n")?;
    let excerpt = EditPredictionExcerpt::select_from_buffer(
        cursor_position,
        active_buffer,
        &zeta2::DEFAULT_EXCERPT_OPTIONS,
        None,
    )
    .context("Can't get cursor excerpt because current line is too long")?;

    write_codeblock(
        &active_full_path,
        &[Excerpt {
            start_line: excerpt.line_range.start,
            text: excerpt.text(active_buffer).body.into(),
        }],
        &[],
        Line(active_buffer.max_point().row),
        true,
        &mut prompt,
    );

    writeln!(&mut prompt, "{TOOL_USE_REMINDER}")?;

    Ok(prompt)
}

fn write_events(output: &mut String, events: VecDeque<Event>) {
    if events.is_empty() {
        return;
    };

    writeln!(output, "`````diff").unwrap();
    for event in events {
        writeln!(output, "{}", event).unwrap();
    }
    writeln!(output, "`````\n").unwrap();
}

const SEARCH_INSTRUCTIONS: &str = indoc! {r#"
    You are part of an edit prediction system in a code editor.
    Your role is to predict the location of the next edit the user will make.

    Analyze the history of edits made by the user in order to infer what they are currently trying to accomplish.
    Then you should use the `search` tool to find code that informs where the next edit should be.

    ## Search instructions

    - Focus on locations in other files that may need to be updated following the users changes. For example:
        - Type declarations that need to be updated to add an field, method, attribute, or implementation
        - Function declarations whose signature needs to be updated to add or change an argument or return type
        - Usages of a module, type, function, field, or variable whose declaration changed in the edit history
        - Configuration files such as database schemas, or dependency lists such as in `Cargo.toml` or `package.json`
        - Import regions that may need to be updated to add a missing import
    - Keep searches as targeted as possible
    - Use `syntax_node` parameter whenever you're looking for a particular type, class, or function
    - Avoid using wildcard globs if you already know the file path of the content you're looking for
    - Always continue along the user's current trajectory, rather than changing course or going back.
"#};

const TOOL_USE_REMINDER: &str = indoc! {"
    --
    Analyze the user's intent in one to two sentences, then call the `search` tool. /no_think
"};

const JUMP_INSTRUCTIONS: &str = indoc! {"
    Now analyze the search results, and explain your findings in 1 or 2 sentences, then if you think another edit is needed,
    output the target file path and line number, like this:

    ```jump
    project/file.rs:123
    ```

    Predict a single location! You can't use any tools beyond this point, output only the target location based on the search results provided. Do not attempt to search again.
"};
