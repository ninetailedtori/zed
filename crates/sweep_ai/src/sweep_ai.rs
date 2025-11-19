mod api;
mod jump;

use anyhow::{Context as _, Result};
use arrayvec::ArrayVec;
use client::telemetry;
use collections::HashMap;
use feature_flags::FeatureFlag;
use futures::AsyncReadExt as _;
use gpui::{App, AppContext, Context, Entity, EntityId, Global, Task, WeakEntity};
use http_client::{AsyncBody, Method};
use language::{
    Anchor, Buffer, BufferSnapshot, EditPreview, Point, ToOffset as _, ToPoint, text_diff,
};
use project::Project;
use release_channel::{AppCommitSha, AppVersion};
use std::collections::{VecDeque, hash_map};
use std::fmt::{self, Display};
use std::mem;
use std::{
    cmp,
    fmt::Write,
    ops::Range,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use util::ResultExt;
use util::rel_path::RelPath;
use workspace::Workspace;

use crate::api::{AutocompleteRequest, AutocompleteResponse, FileChunk};
use crate::jump::predict_jump;

const CHANGE_GROUPING_LINE_SPAN: u32 = 8;
const MAX_EVENT_COUNT: usize = 6;

const SWEEP_API_URL: &str = "https://autocomplete.sweep.dev/backend/next_edit_autocomplete";

pub struct SweepFeatureFlag;

impl FeatureFlag for SweepFeatureFlag {
    const NAME: &str = "sweep-ai";
}

#[derive(Clone)]
struct SweepAiGlobal(Entity<SweepAi>);

impl Global for SweepAiGlobal {}

#[derive(Clone)]
pub struct EditPrediction {
    id: EditPredictionId,
    path: Arc<Path>,
    edits: Arc<[(Range<Anchor>, Arc<str>)]>,
    snapshot: BufferSnapshot,
    edit_preview: EditPreview,
}

impl EditPrediction {
    fn interpolate(&self, new_snapshot: &BufferSnapshot) -> Option<Vec<(Range<Anchor>, Arc<str>)>> {
        edit_prediction::interpolate_edits(&self.snapshot, new_snapshot, &self.edits)
    }
}

impl fmt::Debug for EditPrediction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EditPrediction")
            .field("path", &self.path)
            .field("edits", &self.edits)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct EditPredictionId(String);

impl Display for EditPredictionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct SweepAi {
    projects: HashMap<EntityId, SweepAiProject>,
    debug_info: Arc<str>,
    api_token: Option<String>,
}

struct SweepAiProject {
    events: VecDeque<Event>,
    registered_buffers: HashMap<gpui::EntityId, RegisteredBuffer>,
    current_prediction: Option<CurrentEditPrediction>,
}

impl SweepAi {
    pub fn global(cx: &mut App) -> Option<Entity<Self>> {
        cx.try_global::<SweepAiGlobal>()
            .map(|global| global.0.clone())
    }

    pub fn register(cx: &mut App) -> Entity<Self> {
        Self::global(cx).unwrap_or_else(|| {
            let entity = cx.new(|cx| Self::new(cx));
            cx.set_global(SweepAiGlobal(entity.clone()));
            entity
        })
    }

    pub fn clear_history(&mut self) {
        for sweep_ai_project in self.projects.values_mut() {
            sweep_ai_project.events.clear();
        }
    }

    fn new(cx: &mut Context<Self>) -> Self {
        Self {
            api_token: std::env::var("SWEEP_AI_TOKEN").ok(),
            projects: HashMap::default(),
            debug_info: format!(
                "Zed v{version} ({sha}) - OS: {os} - Zed v{version}",
                version = AppVersion::global(cx),
                sha = AppCommitSha::try_global(cx).map_or("unknown".to_string(), |sha| sha.full()),
                os = telemetry::os_name(),
            )
            .into(),
        }
    }

    fn get_or_init_sweep_ai_project(
        &mut self,
        project: &Entity<Project>,
        cx: &mut Context<Self>,
    ) -> &mut SweepAiProject {
        let project_id = project.entity_id();
        match self.projects.entry(project_id) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => {
                cx.observe_release(project, move |this, _, _cx| {
                    this.projects.remove(&project_id);
                })
                .detach();
                entry.insert(SweepAiProject {
                    events: VecDeque::with_capacity(MAX_EVENT_COUNT),
                    registered_buffers: HashMap::default(),
                    current_prediction: None,
                })
            }
        }
    }

    pub fn register_buffer(
        &mut self,
        buffer: &Entity<Buffer>,
        project: &Entity<Project>,
        cx: &mut Context<Self>,
    ) {
        let sweep_ai_project = self.get_or_init_sweep_ai_project(project, cx);
        Self::register_buffer_impl(sweep_ai_project, buffer, project, cx);
    }

    fn register_buffer_impl<'a>(
        sweep_ai_project: &'a mut SweepAiProject,
        buffer: &Entity<Buffer>,
        project: &Entity<Project>,
        cx: &mut Context<Self>,
    ) -> &'a mut RegisteredBuffer {
        let buffer_id = buffer.entity_id();
        match sweep_ai_project.registered_buffers.entry(buffer_id) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => {
                let snapshot = buffer.read(cx).snapshot();
                let project_entity_id = project.entity_id();
                entry.insert(RegisteredBuffer {
                    snapshot,
                    _subscriptions: [
                        cx.subscribe(buffer, {
                            let project = project.downgrade();
                            move |this, buffer, event, cx| {
                                if let language::BufferEvent::Edited = event
                                    && let Some(project) = project.upgrade()
                                {
                                    this.report_changes_for_buffer(&buffer, &project, cx);
                                }
                            }
                        }),
                        cx.observe_release(buffer, move |this, _buffer, _cx| {
                            let Some(sweep_ai_project) = this.projects.get_mut(&project_entity_id)
                            else {
                                return;
                            };
                            sweep_ai_project.registered_buffers.remove(&buffer_id);
                        }),
                    ],
                })
            }
        }
    }

    fn current_prediction_for_buffer(
        &self,
        buffer: &Entity<Buffer>,
        project: &Entity<Project>,
        cx: &App,
    ) -> Option<BufferEditPrediction<'_>> {
        let project_state = self.projects.get(&project.entity_id())?;

        let CurrentEditPrediction {
            requested_by_buffer_id,
            prediction,
        } = project_state.current_prediction.as_ref()?;

        if prediction.snapshot.remote_id() == buffer.read(cx).remote_id() {
            Some(BufferEditPrediction::Local { prediction })
        } else if *requested_by_buffer_id == buffer.entity_id() {
            Some(BufferEditPrediction::Jump { prediction })
        } else {
            None
        }
    }

    fn discard_current_prediction(&mut self, project: &Entity<Project>) {
        if let Some(project_state) = self.projects.get_mut(&project.entity_id()) {
            project_state.current_prediction.take();
        };
    }

    pub fn refresh_prediction(
        &mut self,
        workspace: &WeakEntity<Workspace>,
        project: &Entity<Project>,
        current_buffer: &Entity<Buffer>,
        position: language::Anchor,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let prediction_task = self.request_prediction(
            workspace,
            project,
            current_buffer,
            current_buffer,
            position,
            cx,
        );
        let project = project.clone();
        let current_buffer_id = current_buffer.entity_id();

        cx.spawn(async move |this, cx| {
            let Some(prediction) = prediction_task.await? else {
                return Ok(());
            };

            this.update(cx, |this, cx| {
                let project_state = this.get_or_init_sweep_ai_project(&project, cx);

                if project_state
                    .current_prediction
                    .as_ref()
                    .is_none_or(|old| old.should_replace_prediction(&old, &prediction.snapshot))
                {
                    project_state.current_prediction = Some(CurrentEditPrediction {
                        requested_by_buffer_id: current_buffer_id,
                        prediction,
                    });
                }
            })
            .ok();

            anyhow::Ok(())
        })
    }

    pub fn request_prediction(
        &mut self,
        workspace: &WeakEntity<Workspace>,
        project: &Entity<Project>,
        current_buffer: &Entity<Buffer>,
        target_buffer: &Entity<Buffer>,
        position: language::Anchor,
        cx: &mut Context<Self>,
    ) -> Task<Result<Option<EditPrediction>>> {
        let snapshot = target_buffer.read(cx).snapshot();
        let debug_info = self.debug_info.clone();
        let Some(api_token) = self.api_token.clone() else {
            return Task::ready(Ok(None));
        };
        let full_path: Arc<Path> = snapshot
            .file()
            .map(|file| file.full_path(cx))
            .unwrap_or_else(|| "untitled".into())
            .into();

        let project_file = project::File::from_dyn(snapshot.file());
        let repo_name = project_file
            .map(|file| file.worktree.read(cx).root_name_str())
            .unwrap_or("untitled")
            .into();
        let offset = position.to_offset(&snapshot);

        let project_state = self.get_or_init_sweep_ai_project(project, cx);
        let events = project_state.events.clone();
        let http_client = cx.http_client();

        let Some(recent_buffers) = workspace
            .read_with(cx, |workspace, cx| {
                workspace
                    .recent_navigation_history_iter(cx)
                    .filter_map(|(project_path, _)| {
                        let buffer = project.read(cx).get_open_buffer(&project_path, cx)?;

                        if target_buffer == &buffer {
                            None
                        } else {
                            Some(buffer.read(cx).snapshot())
                        }
                    })
                    .take(3)
                    .collect::<Vec<_>>()
            })
            .log_err()
        else {
            return Task::ready(Ok(None));
        };

        let result = cx.background_spawn({
            let full_path = full_path.clone();
            // todo! avoid cloning this so much?
            let events = events.clone();
            async move {
                let text = snapshot.text();

                let mut recent_changes = String::new();

                for event in events {
                    writeln!(&mut recent_changes, "{event}")?;
                }

                let file_chunks = recent_buffers
                    .into_iter()
                    .map(|snapshot| {
                        let end_point = language::Point::new(30, 0).min(snapshot.max_point());
                        FileChunk {
                            content: snapshot
                                .text_for_range(language::Point::zero()..end_point)
                                .collect(),
                            file_path: snapshot
                                .file()
                                .map(|f| f.path().as_unix_str())
                                .unwrap_or("untitled")
                                .to_string(),
                            start_line: 0,
                            end_line: end_point.row as usize,
                            timestamp: snapshot.file().and_then(|file| {
                                Some(
                                    file.disk_state()
                                        .mtime()?
                                        .to_seconds_and_nanos_for_persistence()?
                                        .0,
                                )
                            }),
                        }
                    })
                    .collect();

                eprintln!("{recent_changes}");

                let request_body = AutocompleteRequest {
                    debug_info,
                    repo_name,
                    file_path: full_path.clone(),
                    file_contents: text.clone(),
                    original_file_contents: text,
                    cursor_position: offset,
                    recent_changes: recent_changes.clone(),
                    changes_above_cursor: true,
                    multiple_suggestions: false,
                    branch: None,
                    file_chunks,
                    retrieval_chunks: vec![],
                    recent_user_actions: vec![],
                    // TODO
                    privacy_mode_enabled: false,
                };

                dbg!(&request_body.file_contents);
                dbg!(&request_body.cursor_position);

                let mut buf: Vec<u8> = Vec::new();
                let writer = brotli::CompressorWriter::new(&mut buf, 4096, 11, 22);
                serde_json::to_writer(writer, &request_body)?;
                let body: AsyncBody = buf.into();

                let request = http_client::Request::builder()
                    .uri(SWEEP_API_URL)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {}", api_token))
                    .header("Connection", "keep-alive")
                    .header("Content-Encoding", "br")
                    .method(Method::POST)
                    .body(body)?;

                let mut response = http_client.send(request).await?;

                let mut body: Vec<u8> = Vec::new();
                response.body_mut().read_to_end(&mut body).await?;

                if !response.status().is_success() {
                    anyhow::bail!(
                        "Request failed with status: {:?}\nBody: {}",
                        response.status(),
                        String::from_utf8_lossy(&body),
                    );
                };

                let response: AutocompleteResponse = serde_json::from_slice(&body)?;
                dbg!(&response);

                let old_text = snapshot
                    .text_for_range(response.start_index..response.end_index)
                    .collect::<String>();
                let edits = text_diff(&old_text, &response.completion)
                    .into_iter()
                    .map(|(range, text)| {
                        (
                            snapshot.anchor_after(response.start_index + range.start)
                                ..snapshot.anchor_before(response.start_index + range.end),
                            text,
                        )
                    })
                    .collect::<Vec<_>>();

                anyhow::Ok((response.autocomplete_id, edits, snapshot))
            }
        });

        let current_buffer = current_buffer.clone();
        let target_buffer = target_buffer.clone();
        let project = project.clone();
        let workspace = workspace.clone();

        cx.spawn(async move |this, cx| {
            let (id, edits, old_snapshot) = result.await?;

            if edits.is_empty() && current_buffer == target_buffer {
                let predict_jump_task = current_buffer.update(cx, |buffer, cx| {
                    let snapshot = buffer.snapshot();
                    let cursor_point = position.to_point(&snapshot);

                    predict_jump(
                        full_path,
                        snapshot,
                        cursor_point,
                        project.clone(),
                        events,
                        cx,
                    )
                })?;

                if let Some(jump_target) = predict_jump_task.await? {
                    let prediction = this
                        .update(cx, |this, cx| {
                            this.request_prediction(
                                &workspace,
                                &project,
                                &current_buffer,
                                &jump_target.buffer,
                                jump_target.anchor,
                                cx,
                            )
                        })?
                        .await;

                    return prediction;
                }

                return anyhow::Ok(None);
            }

            let Some((edits, new_snapshot, preview_task)) =
                target_buffer.read_with(cx, |buffer, cx| {
                    let new_snapshot = buffer.snapshot();

                    let edits: Arc<[(Range<Anchor>, Arc<str>)]> =
                        edit_prediction::interpolate_edits(&old_snapshot, &new_snapshot, &edits)?
                            .into();
                    let preview_task = buffer.preview_edits(edits.clone(), cx);

                    Some((edits, new_snapshot, preview_task))
                })?
            else {
                return anyhow::Ok(None);
            };

            let prediction = EditPrediction {
                id: EditPredictionId(id),
                path: full_path,
                edits,
                snapshot: new_snapshot,
                edit_preview: preview_task.await,
            };

            anyhow::Ok(Some(prediction))
        })
    }

    fn report_changes_for_buffer(
        &mut self,
        buffer: &Entity<Buffer>,
        project: &Entity<Project>,
        cx: &mut Context<Self>,
    ) {
        let sweep_ai_project = self.get_or_init_sweep_ai_project(project, cx);
        let registered_buffer = Self::register_buffer_impl(sweep_ai_project, buffer, project, cx);

        let new_snapshot = buffer.read(cx).snapshot();
        if new_snapshot.version == registered_buffer.snapshot.version {
            return;
        }

        let old_snapshot = mem::replace(&mut registered_buffer.snapshot, new_snapshot.clone());
        let end_edit_anchor = new_snapshot
            .anchored_edits_since::<Point>(&old_snapshot.version)
            .last()
            .map(|(_, range)| range.end);
        let events = &mut sweep_ai_project.events;

        if let Some(Event::BufferChange {
            new_snapshot: last_new_snapshot,
            end_edit_anchor: last_end_edit_anchor,
            ..
        }) = events.back_mut()
        {
            let is_next_snapshot_of_same_buffer = old_snapshot.remote_id()
                == last_new_snapshot.remote_id()
                && old_snapshot.version == last_new_snapshot.version;

            let should_coalesce = is_next_snapshot_of_same_buffer
                && end_edit_anchor
                    .as_ref()
                    .zip(last_end_edit_anchor.as_ref())
                    .is_some_and(|(a, b)| {
                        let a = a.to_point(&new_snapshot);
                        let b = b.to_point(&new_snapshot);
                        a.row.abs_diff(b.row) <= CHANGE_GROUPING_LINE_SPAN
                    });

            if should_coalesce {
                *last_end_edit_anchor = end_edit_anchor.clone();
                *last_new_snapshot = new_snapshot.clone();
                return;
            }
        }

        if events.len() >= MAX_EVENT_COUNT {
            events.pop_front();
        }

        events.push_back(Event::BufferChange {
            old_snapshot,
            new_snapshot: new_snapshot.clone(),
            end_edit_anchor,
        });
    }
}

struct RegisteredBuffer {
    snapshot: BufferSnapshot,
    _subscriptions: [gpui::Subscription; 2],
}

#[derive(Clone)]
pub enum Event {
    BufferChange {
        old_snapshot: BufferSnapshot,
        new_snapshot: BufferSnapshot,
        end_edit_anchor: Option<Anchor>,
    },
}

impl Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Event::BufferChange {
                old_snapshot,
                new_snapshot,
                ..
            } => {
                let old_path = old_snapshot
                    .file()
                    .map(|f| f.path().as_ref())
                    .unwrap_or(RelPath::unix("untitled").unwrap());
                let new_path = new_snapshot
                    .file()
                    .map(|f| f.path().as_ref())
                    .unwrap_or(RelPath::unix("untitled").unwrap());
                if old_path != new_path {
                    // TODO confirm how to do this for sweep
                    // writeln!(f, "User renamed {:?} to {:?}\n", old_path, new_path)?;
                }

                let diff = language::unified_diff(&old_snapshot.text(), &new_snapshot.text());
                if !diff.is_empty() {
                    write!(
                        f,
                        "File: {}:\n{}\n",
                        new_path.display(util::paths::PathStyle::Posix),
                        diff
                    )?
                }

                fmt::Result::Ok(())
            }
        }
    }
}

struct CurrentEditPrediction {
    requested_by_buffer_id: EntityId,
    prediction: EditPrediction,
}

impl CurrentEditPrediction {
    // todo! rename completion -> prediction
    fn should_replace_prediction(&self, old_completion: &Self, snapshot: &BufferSnapshot) -> bool {
        if self.requested_by_buffer_id != old_completion.requested_by_buffer_id {
            return true;
        }

        let Some(old_edits) = old_completion.prediction.interpolate(snapshot) else {
            return true;
        };
        let Some(new_edits) = self.prediction.interpolate(snapshot) else {
            return false;
        };

        if old_edits.len() == 1 && new_edits.len() == 1 {
            let (old_range, old_text) = &old_edits[0];
            let (new_range, new_text) = &new_edits[0];
            new_range == old_range && new_text.starts_with(old_text.as_ref())
        } else {
            true
        }
    }
}

/// A prediction from the perspective of a buffer.
#[derive(Debug)]
enum BufferEditPrediction<'a> {
    Local { prediction: &'a EditPrediction },
    Jump { prediction: &'a EditPrediction },
}

struct PendingCompletion {
    id: usize,
    _task: Task<()>,
}

pub struct SweepAiEditPredictionProvider {
    workspace: WeakEntity<Workspace>,
    sweep_ai: Entity<SweepAi>,
    pending_completions: ArrayVec<PendingCompletion, 2>,
    next_pending_completion_id: usize,
    last_request_timestamp: Instant,
    project: Entity<Project>,
}

impl SweepAiEditPredictionProvider {
    pub const THROTTLE_TIMEOUT: Duration = Duration::from_millis(300);

    pub fn new(
        sweep_ai: Entity<SweepAi>,
        workspace: WeakEntity<Workspace>,
        project: Entity<Project>,
    ) -> Self {
        Self {
            sweep_ai,
            pending_completions: ArrayVec::new(),
            next_pending_completion_id: 0,
            last_request_timestamp: Instant::now(),
            project,
            workspace,
        }
    }
}

impl edit_prediction::EditPredictionProvider for SweepAiEditPredictionProvider {
    fn name() -> &'static str {
        "zed-predict"
    }

    fn display_name() -> &'static str {
        "Zed's Edit Predictions"
    }

    fn show_completions_in_menu() -> bool {
        true
    }

    fn show_tab_accept_marker() -> bool {
        true
    }

    fn is_enabled(
        &self,
        _buffer: &Entity<Buffer>,
        _cursor_position: language::Anchor,
        cx: &App,
    ) -> bool {
        self.sweep_ai.read(cx).api_token.is_some()
    }

    fn is_refreshing(&self) -> bool {
        !self.pending_completions.is_empty()
    }

    fn refresh(
        &mut self,
        buffer: Entity<Buffer>,
        position: language::Anchor,
        _debounce: bool,
        cx: &mut Context<Self>,
    ) {
        let sweep_ai = self.sweep_ai.read(cx);

        if let Some(current) = sweep_ai.current_prediction_for_buffer(&buffer, &self.project, cx)
            && let BufferEditPrediction::Local { prediction } = current
            && prediction
                .interpolate(&buffer.read(cx).snapshot())
                .is_some()
        {
            return;
        }

        let pending_completion_id = self.next_pending_completion_id;
        self.next_pending_completion_id += 1;
        let last_request_timestamp = self.last_request_timestamp;

        let project = self.project.clone();
        let workspace = self.workspace.clone();
        let task = cx.spawn(async move |this, cx| {
            if let Some(timeout) = (last_request_timestamp + Self::THROTTLE_TIMEOUT)
                .checked_duration_since(Instant::now())
            {
                cx.background_executor().timer(timeout).await;
            }

            let refresh_task = this.update(cx, |this, cx| {
                this.last_request_timestamp = Instant::now();
                this.sweep_ai.update(cx, |sweep_ai, cx| {
                    sweep_ai.refresh_prediction(&workspace, &project, &buffer, position, cx)
                })
            });

            if let Some(refresh_task) = refresh_task.ok() {
                refresh_task.await.log_err();
            }

            this.update(cx, |this, cx| {
                if this.pending_completions[0].id == pending_completion_id {
                    this.pending_completions.remove(0);
                } else {
                    this.pending_completions.clear();
                }

                cx.notify();
            })
            .ok();
        });

        // We always maintain at most two pending completions. When we already
        // have two, we replace the newest one.
        if self.pending_completions.len() <= 1 {
            self.pending_completions.push(PendingCompletion {
                id: pending_completion_id,
                _task: task,
            });
        } else if self.pending_completions.len() == 2 {
            self.pending_completions.pop();
            self.pending_completions.push(PendingCompletion {
                id: pending_completion_id,
                _task: task,
            });
        }
    }

    fn cycle(
        &mut self,
        _buffer: Entity<Buffer>,
        _cursor_position: language::Anchor,
        _direction: edit_prediction::Direction,
        _cx: &mut Context<Self>,
    ) {
        // Right now we don't support cycling.
    }

    fn accept(&mut self, _cx: &mut Context<Self>) {
        self.pending_completions.clear();
    }

    fn discard(&mut self, cx: &mut Context<Self>) {
        self.pending_completions.clear();
        self.sweep_ai.update(cx, |sweep_ai, _cx| {
            sweep_ai.discard_current_prediction(&self.project);
        });
    }

    fn suggest(
        &mut self,
        buffer: &Entity<Buffer>,
        cursor_position: language::Anchor,
        cx: &mut Context<Self>,
    ) -> Option<edit_prediction::EditPrediction> {
        let prediction =
            self.sweep_ai
                .read(cx)
                .current_prediction_for_buffer(buffer, &self.project, cx)?;

        let prediction = match prediction {
            BufferEditPrediction::Local { prediction } => prediction,
            BufferEditPrediction::Jump { prediction } => {
                return Some(edit_prediction::EditPrediction::Jump {
                    id: Some(prediction.id.to_string().into()),
                    snapshot: prediction.snapshot.clone(),
                    target: prediction.edits.first().unwrap().0.start,
                });
            }
        };

        let buffer = buffer.read(cx);
        let Some(edits) = prediction.interpolate(&buffer.snapshot()) else {
            self.sweep_ai.update(cx, |sweep_ai, _cx| {
                sweep_ai.discard_current_prediction(&self.project);
            });
            return None;
        };

        let cursor_row = cursor_position.to_point(buffer).row;
        let (closest_edit_ix, (closest_edit_range, _)) =
            edits.iter().enumerate().min_by_key(|(_, (range, _))| {
                let distance_from_start = cursor_row.abs_diff(range.start.to_point(buffer).row);
                let distance_from_end = cursor_row.abs_diff(range.end.to_point(buffer).row);
                cmp::min(distance_from_start, distance_from_end)
            })?;

        let mut edit_start_ix = closest_edit_ix;
        for (range, _) in edits[..edit_start_ix].iter().rev() {
            let distance_from_closest_edit =
                closest_edit_range.start.to_point(buffer).row - range.end.to_point(buffer).row;
            if distance_from_closest_edit <= 1 {
                edit_start_ix -= 1;
            } else {
                break;
            }
        }

        let mut edit_end_ix = closest_edit_ix + 1;
        for (range, _) in &edits[edit_end_ix..] {
            let distance_from_closest_edit =
                range.start.to_point(buffer).row - closest_edit_range.end.to_point(buffer).row;
            if distance_from_closest_edit <= 1 {
                edit_end_ix += 1;
            } else {
                break;
            }
        }

        Some(edit_prediction::EditPrediction::Local {
            id: Some(prediction.id.to_string().into()),
            edits: edits[edit_start_ix..edit_end_ix].to_vec(),
            edit_preview: Some(prediction.edit_preview.clone()),
        })
    }
}
