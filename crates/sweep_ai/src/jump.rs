use anyhow::{Context as _, Result};
use edit_prediction_context::{EditPredictionExcerpt, EditPredictionExcerptOptions, Line};
use gpui::Entity;
use indoc::indoc;
use language::{Anchor, Buffer, BufferSnapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{fmt::Write, path::PathBuf};

use cloud_zeta2_prompt::{Excerpt, push_events, write_codeblock};

use crate::Event;

pub fn build_jump_prompt(
    active_full_path: PathBuf,
    active_buffer: &BufferSnapshot,
    cursor_position: language::Point,
    events: &[Event],
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

fn write_events(output: &mut String, events: &[Event]) {
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
    Analyze the user's intent in one to two sentences, then call the `search` tool.
"};
