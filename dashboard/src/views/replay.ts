import type { DashboardData } from "../types";
import { navigate } from "../router";
import { prettyJson } from "../utils/format";

export function renderReplay(
  container: HTMLElement,
  data: DashboardData,
  params: Record<string, string>,
): void {
  const runIds = Object.keys(data.runs);
  const selectedRunId =
    params.runId && data.runs[params.runId] ? params.runId : runIds[0];
  const caseIds = data.case_order;
  const selectedCaseId =
    params.caseId && data.cases[params.caseId] ? params.caseId : caseIds[0];

  // -- Controls --
  const controls = document.createElement("div");
  controls.className = "view-controls";

  // Run selector
  const runLabel = document.createElement("label");
  runLabel.textContent = "Run: ";

  const runSelect = document.createElement("select");
  for (const rid of runIds) {
    const opt = document.createElement("option");
    opt.value = rid;
    opt.textContent = data.runs[rid].label;
    if (rid === selectedRunId) opt.selected = true;
    runSelect.appendChild(opt);
  }
  runLabel.appendChild(runSelect);
  controls.appendChild(runLabel);

  // Case selector (grouped by category)
  const caseLabel = document.createElement("label");
  caseLabel.textContent = "Case: ";

  const caseSelect = document.createElement("select");

  // Group cases by category
  const categorized = new Map<string, string[]>();
  for (const cid of caseIds) {
    const cat = data.cases[cid].category;
    if (!categorized.has(cat)) categorized.set(cat, []);
    categorized.get(cat)!.push(cid);
  }

  for (const [category, ids] of categorized) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = `Category ${category}`;
    for (const cid of ids) {
      const opt = document.createElement("option");
      opt.value = cid;
      opt.textContent = `${cid} - ${data.cases[cid].title}`;
      if (cid === selectedCaseId) opt.selected = true;
      optgroup.appendChild(opt);
    }
    caseSelect.appendChild(optgroup);
  }

  caseLabel.appendChild(caseSelect);
  controls.appendChild(caseLabel);
  container.appendChild(controls);

  // Navigation on change
  runSelect.addEventListener("change", () => {
    navigate(`/replay/${runSelect.value}/${caseSelect.value}`);
  });
  caseSelect.addEventListener("change", () => {
    navigate(`/replay/${runSelect.value}/${caseSelect.value}`);
  });

  // -- Chat container --
  const chatContainer = document.createElement("div");
  chatContainer.className = "chat-container";
  chatContainer.style.marginTop = "1rem";
  container.appendChild(chatContainer);

  // Render messages
  const run = data.runs[selectedRunId];
  if (!run) {
    chatContainer.textContent = "Run not found.";
    return;
  }

  const conversation = run.conversations[selectedCaseId];
  if (!conversation || !conversation.messages || conversation.messages.length === 0) {
    chatContainer.textContent = "No conversation data for this case.";
    return;
  }

  for (const msg of conversation.messages) {
    switch (msg.role) {
      case "system":
        renderSystemBubble(chatContainer, msg.content);
        break;
      case "user":
        renderUserBubble(chatContainer, msg.content);
        break;
      case "assistant":
        renderAssistantBubble(chatContainer, msg);
        break;
      case "tool":
        renderToolResultBubble(chatContainer, msg.content);
        break;
    }
  }
}

// ---------- Bubble renderers ----------

function renderSystemBubble(container: HTMLElement, content: string | null): void {
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble system";

  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = "System Prompt";
  bubble.appendChild(label);

  const body = document.createElement("div");
  body.className = "bubble-content";
  body.textContent = content || "";
  bubble.appendChild(body);

  bubble.addEventListener("click", () => {
    bubble.classList.toggle("expanded");
  });

  container.appendChild(bubble);
}

function renderUserBubble(container: HTMLElement, content: string | null): void {
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble user";

  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = "User";
  bubble.appendChild(label);

  const body = document.createElement("div");
  body.className = "bubble-content";
  body.textContent = content || "";
  bubble.appendChild(body);

  container.appendChild(bubble);
}

function renderAssistantBubble(
  container: HTMLElement,
  msg: { content: string | null; reasoning_content?: string; tool_calls?: Array<{ type: string; function: { name: string; arguments: string }; id: string }> },
): void {
  // Thinking block (collapsible)
  if (msg.reasoning_content) {
    const thinkingBubble = document.createElement("div");
    thinkingBubble.className = "chat-bubble assistant";

    const thinkingBlock = document.createElement("div");
    thinkingBlock.className = "thinking-block";

    const thinkingLabel = document.createElement("div");
    thinkingLabel.className = "bubble-label";
    thinkingLabel.textContent = "Thinking";
    thinkingBlock.appendChild(thinkingLabel);

    const thinkingContent = document.createElement("div");
    thinkingContent.className = "thinking-content";
    thinkingContent.textContent = msg.reasoning_content;
    thinkingBlock.appendChild(thinkingContent);

    thinkingBlock.addEventListener("click", () => {
      thinkingBlock.classList.toggle("expanded");
    });

    thinkingBubble.appendChild(thinkingBlock);
    container.appendChild(thinkingBubble);
  }

  // Tool calls — each as a separate bubble
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    for (const tc of msg.tool_calls) {
      const tcBubble = document.createElement("div");
      tcBubble.className = "chat-bubble tool-call";

      const label = document.createElement("div");
      label.className = "bubble-label";
      label.textContent = "Tool Call";
      tcBubble.appendChild(label);

      const nameEl = document.createElement("div");
      nameEl.style.fontWeight = "600";
      nameEl.style.marginBottom = "0.25rem";
      nameEl.textContent = tc.function.name;
      tcBubble.appendChild(nameEl);

      const argsBlock = document.createElement("div");
      argsBlock.className = "code-block";
      try {
        argsBlock.textContent = prettyJson(JSON.parse(tc.function.arguments));
      } catch {
        argsBlock.textContent = tc.function.arguments;
      }
      tcBubble.appendChild(argsBlock);

      container.appendChild(tcBubble);
    }
  }

  // Text content
  if (msg.content) {
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble assistant";

    const label = document.createElement("div");
    label.className = "bubble-label";
    label.textContent = "Assistant";
    bubble.appendChild(label);

    const body = document.createElement("div");
    body.className = "bubble-content";
    body.textContent = msg.content;
    bubble.appendChild(body);

    container.appendChild(bubble);
  }
}

function renderToolResultBubble(container: HTMLElement, content: string | null): void {
  // Determine if this is a submit_response result
  let isSubmit = false;
  let parsed: unknown = null;

  if (content) {
    try {
      parsed = JSON.parse(content);
      if (
        parsed &&
        typeof parsed === "object" &&
        "accepted" in (parsed as Record<string, unknown>)
      ) {
        isSubmit = true;
      }
    } catch {
      // Not JSON, render as-is
    }
  }

  const bubble = document.createElement("div");
  bubble.className = isSubmit ? "chat-bubble submit" : "chat-bubble tool-result";

  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = isSubmit ? "Submit Response" : "Tool Result";
  bubble.appendChild(label);

  const codeBlock = document.createElement("div");
  codeBlock.className = "code-block";
  if (parsed !== null) {
    codeBlock.textContent = prettyJson(parsed);
  } else {
    codeBlock.textContent = content || "";
  }
  bubble.appendChild(codeBlock);

  container.appendChild(bubble);
}
