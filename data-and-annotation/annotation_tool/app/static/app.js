const state = {
  sessionToken: localStorage.getItem("vocalgrad_session_token") || "",
  user: null,
  currentItem: null,
  currentProgress: null,
  selectedLabel: null,
  playCount: 0,
  startedAt: null,
};

const loginScreen = document.getElementById("login-screen");
const annotationScreen = document.getElementById("annotation-screen");
const categoryTransitionScreen = document.getElementById("category-transition-screen");
const completionScreen = document.getElementById("completion-screen");
const adminScreen = document.getElementById("admin-screen");

const userIdInput = document.getElementById("user-id");
const passwordInput = document.getElementById("password");
const loginButton = document.getElementById("login-button");
const logoutButtons = [
  document.getElementById("logout-button"),
  document.getElementById("logout-button-transition"),
  document.getElementById("logout-button-complete"),
  document.getElementById("logout-button-admin"),
].filter(Boolean);

const loginError = document.getElementById("login-error");
const annotationError = document.getElementById("annotation-error");
const adminError = document.getElementById("admin-error");

const annotatorTitle = document.getElementById("annotator-title");
const annotatorSubtitle = document.getElementById("annotator-subtitle");
const categoryProgress = document.getElementById("category-progress");
const overallProgress = document.getElementById("overall-progress");
const promptText = document.getElementById("prompt-text");
const playButton = document.getElementById("play-button");
const audioElement = document.getElementById("audio-element");
const submitButton = document.getElementById("submit-button");
const answerButtons = Array.from(document.querySelectorAll(".answer-button"));
const summaryRoot = document.getElementById("summary-root");
const completionSubtitle = document.getElementById("completion-subtitle");
const transitionTitle = document.getElementById("transition-title");
const transitionSubtitle = document.getElementById("transition-subtitle");
const transitionMessage = document.getElementById("transition-message");
const transitionContinueButton = document.getElementById("transition-continue-button");

const adminSubtitle = document.getElementById("admin-subtitle");
const adminRoot = document.getElementById("admin-root");
const refreshAdminButton = document.getElementById("refresh-admin-button");
const bulkDownloadButton = document.getElementById("bulk-download-button");
const adminSummaryMode = document.getElementById("admin-summary-mode");
const adminSummaryValue = document.getElementById("admin-summary-value");
const resetUserSelect = document.getElementById("reset-user-select");
const resetUserButton = document.getElementById("reset-user-button");

function showError(node, message) {
  node.textContent = message;
  node.classList.remove("hidden");
}

function hideError(node) {
  node.textContent = "";
  node.classList.add("hidden");
}

function showScreen(screen) {
  [loginScreen, annotationScreen, categoryTransitionScreen, completionScreen, adminScreen].forEach((node) => {
    node.classList.toggle("hidden", node !== screen);
  });
}

function setSessionToken(token) {
  state.sessionToken = token || "";
  if (state.sessionToken) {
    localStorage.setItem("vocalgrad_session_token", state.sessionToken);
  } else {
    localStorage.removeItem("vocalgrad_session_token");
  }
}

async function fetchJson(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (!headers.has("Content-Type") && options.body) {
    headers.set("Content-Type", "application/json");
  }
  if (state.sessionToken) {
    headers.set("Authorization", `Bearer ${state.sessionToken}`);
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

function renderSummary(summary) {
  const byGroupTable = (title, data) => {
    const rows = Object.entries(data || {})
      .map(
        ([key, value]) => `
          <tr>
            <td>${key}</td>
            <td>${value.correct}/${value.total}</td>
            <td>${(value.accuracy * 100).toFixed(1)}%</td>
          </tr>
        `,
      )
      .join("");
    return `
      <h3>${title}</h3>
      <table class="summary-table">
        <thead>
          <tr><th>Group</th><th>Correct</th><th>Accuracy</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  };

  summaryRoot.innerHTML = `
    <p><strong>User ID:</strong> ${summary.annotator_id}</p>
    <p><strong>Total completed items:</strong> ${summary.total_completed_items}</p>
    <p><strong>Overall accuracy:</strong> ${(summary.overall_accuracy * 100).toFixed(1)}%</p>
    <p><strong>Average replay count:</strong> ${summary.average_replay_count.toFixed(2)}</p>
    <p><strong>Average response time:</strong> ${summary.average_response_time_ms.toFixed(1)} ms</p>
    ${byGroupTable("Accuracy By Category", summary.accuracy_by_category)}
    ${byGroupTable("Accuracy By Difficulty", summary.accuracy_by_difficulty)}
    ${byGroupTable("Accuracy By Trajectory", summary.accuracy_by_trajectory)}
  `;
}

function renderAnnotationState(payload) {
  const { user, state: sessionState } = payload;
  state.user = user;
  state.currentItem = sessionState.current_item;
  state.currentProgress = sessionState.current_item ? sessionState.current_item.progress : null;
  state.selectedLabel = null;
  state.playCount = 0;
  state.startedAt = new Date().toISOString();
  submitButton.disabled = true;
  answerButtons.forEach((button) => button.classList.remove("selected"));

  if (sessionState.is_complete) {
    completionSubtitle.textContent = `${user.user_id} completed ${sessionState.overall_progress.total_items} trials.`;
    renderSummary(payload.summary);
    showScreen(completionScreen);
    return;
  }

  const item = sessionState.current_item;
  annotatorTitle.textContent = `Category ${item.progress.category_position} of ${item.progress.category_count}`;
  annotatorSubtitle.textContent = `${user.user_id} · ${item.progress.completed_in_category}/${item.progress.total_in_category} items in current category`;
  categoryProgress.textContent = `${item.progress.completed_in_category} / ${item.progress.total_in_category}`;
  overallProgress.textContent = `${item.progress.completed_overall} / ${item.progress.total_overall}`;
  promptText.textContent =
    `Does the ${item.attribute} increase or decrease over time?\n\n` +
    'Answer with only one word: "increase" or "decrease".';
  audioElement.src = item.audio_url;
  audioElement.currentTime = 0;

  hideError(annotationError);
  showScreen(annotationScreen);
}

function renderCategoryTransition(item, user) {
  state.currentItem = item;
  state.currentProgress = item.progress;
  state.selectedLabel = null;
  state.playCount = 0;
  state.startedAt = null;
  transitionTitle.textContent = `Category ${item.progress.category_position} of ${item.progress.category_count}`;
  transitionSubtitle.textContent = `${user.user_id} · ${item.progress.total_in_category} items in this category`;
  transitionMessage.textContent =
    "The category has changed. Please review the instruction carefully before continuing.\n\n" +
    `You will now annotate whether the ${item.attribute} increases or decreases over time.`;
  showScreen(categoryTransitionScreen);
}

function renderAdminTable(title, data) {
  const rows = Object.entries(data || {})
    .map(
      ([key, value]) => `
        <tr>
          <td>${key}</td>
          <td>${value.total_completed_items}</td>
          <td>${(value.overall_accuracy * 100).toFixed(1)}%</td>
        </tr>
      `,
    )
    .join("");
  return `
    <h3>${title}</h3>
    <table class="summary-table">
      <thead>
        <tr><th>Name</th><th>Items</th><th>Accuracy</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function buildCategoryRows(dashboard, mode, value) {
  return Object.entries(dashboard.overall_by_category || {})
    .map(([category, summary]) => {
      if (mode === "difficulty") {
        const group = (summary.accuracy_by_difficulty || {})[value];
        return `
          <tr>
            <td>${category}</td>
            <td>${group ? group.total : 0}</td>
            <td>${group ? (group.accuracy * 100).toFixed(1) : "0.0"}%</td>
          </tr>
        `;
      }
      if (mode === "trajectory") {
        const group = (summary.accuracy_by_trajectory || {})[value];
        return `
          <tr>
            <td>${category}</td>
            <td>${group ? group.total : 0}</td>
            <td>${group ? (group.accuracy * 100).toFixed(1) : "0.0"}%</td>
          </tr>
        `;
      }
      return `
        <tr>
          <td>${category}</td>
          <td>${summary.total_completed_items}</td>
          <td>${(summary.overall_accuracy * 100).toFixed(1)}%</td>
        </tr>
      `;
    })
    .join("");
}

function updateAdminSummaryControls() {
  const mode = adminSummaryMode.value;
  const options =
    mode === "difficulty"
      ? ["low", "mid", "high"]
      : mode === "trajectory"
        ? ["linear", "quad_first_flat", "quad_last_flat", "jump"]
        : [];

  if (options.length === 0) {
    adminSummaryValue.innerHTML = "";
    adminSummaryValue.classList.add("hidden");
    return;
  }

  const current = adminSummaryValue.value;
  adminSummaryValue.innerHTML = options
    .map((value) => `<option value="${value}">${value}</option>`)
    .join("");
  if (options.includes(current)) {
    adminSummaryValue.value = current;
  }
  adminSummaryValue.classList.remove("hidden");
}

function renderAdminDashboard(dashboard) {
  updateAdminSummaryControls();
  const overallRows = buildCategoryRows(
    dashboard,
    adminSummaryMode.value || "overall",
    adminSummaryValue.value || "",
  );

  const agreementRows = Object.entries(dashboard.agreement_by_category || {})
    .map(
      ([category, summary]) => `
        <tr>
          <td>${category}</td>
          <td>${summary.num_items_with_full_overlap}</td>
          <td>${summary.item_level_mean_agreement == null ? "-" : (summary.item_level_mean_agreement * 100).toFixed(1) + "%"}</td>
        </tr>
      `,
    )
    .join("");

  const annotatorBlocks = (dashboard.annotators || [])
    .map(
      ({ user, overall_summary }) => `
        <div class="admin-card">
          <h4>${user.user_id}</h4>
          <p>${user.category_order.join(" → ")}</p>
          <p>${overall_summary.total_completed_items} items · ${(overall_summary.overall_accuracy * 100).toFixed(1)}%</p>
        </div>
      `,
    )
    .join("");

  const testBlocks = (dashboard.test_users || [])
    .map(
      ({ user, overall_summary }) => `
        <div class="admin-card">
          <h4>${user.user_id}</h4>
          <p>${user.category_order.join(" → ")}</p>
          <p>${overall_summary.total_completed_items} items · ${(overall_summary.overall_accuracy * 100).toFixed(1)}%</p>
        </div>
      `,
    )
    .join("");

  const downloadRows = (dashboard.raw_downloads || [])
    .map(
      (entry) => `
        <tr>
          <td>${entry.category}</td>
          <td>${entry.annotator_id}</td>
          <td><button type="button" class="download-button" data-category="${entry.category}" data-annotator="${entry.annotator_id}">Download</button></td>
        </tr>
      `,
    )
    .join("");

  const resettableOptions = (dashboard.resettable_users || [])
    .map(
      (user) => `
        <option value="${user.user_id}">
          ${user.user_id} (${user.role})
        </option>
      `,
    )
    .join("");

  adminSubtitle.textContent = `Credentials file: ${dashboard.credentials_path}`;
  resetUserSelect.innerHTML = resettableOptions;
  adminRoot.innerHTML = `
    <h3>Overall By Category</h3>
    <table class="summary-table">
      <thead>
        <tr><th>Category</th><th>Items</th><th>Accuracy</th></tr>
      </thead>
      <tbody>${overallRows}</tbody>
    </table>

    <h3>Agreement By Category</h3>
    <table class="summary-table">
      <thead>
        <tr><th>Category</th><th>Full Overlap Items</th><th>Mean Agreement</th></tr>
      </thead>
      <tbody>${agreementRows}</tbody>
    </table>

    <h3>Annotators</h3>
    <div class="card-grid">${annotatorBlocks}</div>

    <h3>Test User</h3>
    <div class="card-grid">${testBlocks || "<p class='muted'>No test user data.</p>"}</div>

    <h3>Raw Annotation Files</h3>
    <table class="summary-table">
      <thead>
        <tr><th>Category</th><th>User</th><th>Download</th></tr>
      </thead>
      <tbody>${downloadRows}</tbody>
    </table>
  `;

  adminRoot.querySelectorAll(".download-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const category = button.dataset.category;
      const annotatorId = button.dataset.annotator;
      try {
        const response = await fetch(
          `/api/admin/download/raw?category=${encodeURIComponent(category)}&annotator_id=${encodeURIComponent(annotatorId)}`,
          {
            headers: {
              Authorization: `Bearer ${state.sessionToken}`,
            },
          },
        );
        if (!response.ok) {
          throw new Error("Download failed.");
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${annotatorId}.jsonl`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      } catch (error) {
        showError(adminError, error.message);
      }
    });
  });

  hideError(adminError);
  showScreen(adminScreen);
}

async function loadSessionState() {
  if (!state.sessionToken) {
    showScreen(loginScreen);
    return;
  }
  try {
    const payload = await fetchJson("/api/session/state");
    state.user = payload.user;
    if (payload.user.role === "admin") {
      renderAdminDashboard(payload.dashboard);
      return;
    }
    const progress = payload.state.current_item?.progress;
    if (progress && progress.completed_in_category === 0) {
      renderCategoryTransition(payload.state.current_item, payload.user);
      return;
    }
    renderAnnotationState(payload);
  } catch (_error) {
    setSessionToken("");
    showScreen(loginScreen);
  }
}

async function login() {
  hideError(loginError);
  try {
    const payload = await fetchJson("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({
        user_id: userIdInput.value,
        password: passwordInput.value,
      }),
    });
    setSessionToken(payload.session_token);
    state.user = payload.user;

    if (payload.user.role === "admin") {
      renderAdminDashboard(payload.dashboard);
      return;
    }

    const nextPayload = {
      user: payload.user,
      state: payload.state,
      summary: payload.summary,
    };
    const progress = payload.state.current_item?.progress;
    if (progress && progress.completed_in_category === 0) {
      renderCategoryTransition(payload.state.current_item, payload.user);
      return;
    }
    renderAnnotationState(nextPayload);
  } catch (error) {
    showError(loginError, error.message);
  }
}

async function logout() {
  try {
    await fetchJson("/api/auth/logout", { method: "POST" });
  } catch (_error) {
    // Ignore logout errors and clear local state anyway.
  }
  setSessionToken("");
  state.user = null;
  state.currentItem = null;
  state.selectedLabel = null;
  audioElement.pause();
  audioElement.src = "";
  passwordInput.value = "";
  showScreen(loginScreen);
}

loginButton.addEventListener("click", login);
passwordInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    login();
  }
});
logoutButtons.forEach((button) => button.addEventListener("click", logout));

playButton.addEventListener("click", async () => {
  if (!state.currentItem) {
    return;
  }
  state.playCount += 1;
  audioElement.currentTime = 0;
  try {
    await audioElement.play();
  } catch (_error) {
    showError(annotationError, "Unable to play audio.");
  }
});

answerButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.selectedLabel = button.dataset.label;
    submitButton.disabled = false;
    answerButtons.forEach((other) => {
      other.classList.toggle("selected", other === button);
    });
  });
});

submitButton.addEventListener("click", async () => {
  if (!state.currentItem || !state.selectedLabel) {
    return;
  }

  hideError(annotationError);
  submitButton.disabled = true;
  const submittedAt = new Date().toISOString();
  const responseTimeMs = Math.max(
    0,
    new Date(submittedAt).getTime() - new Date(state.startedAt).getTime(),
  );

  try {
    const payload = await fetchJson("/api/annotation/submit", {
      method: "POST",
      body: JSON.stringify({
        category: state.currentItem.category,
        item_id: state.currentItem.item_id,
        annotated_label: state.selectedLabel,
        response_time_ms: responseTimeMs,
        replay_count: Math.max(state.playCount - 1, 0),
        started_at: state.startedAt,
        submitted_at: submittedAt,
      }),
    });
    const nextPayload = {
      user: state.user,
      state: payload.state,
      summary: payload.summary,
    };
    const progress = payload.state.current_item?.progress;
    if (progress && progress.completed_in_category === 0) {
      renderCategoryTransition(payload.state.current_item, state.user);
      return;
    }
    renderAnnotationState(nextPayload);
  } catch (error) {
    submitButton.disabled = false;
    showError(annotationError, error.message);
  }
});

transitionContinueButton.addEventListener("click", () => {
  if (!state.currentItem || !state.user) {
    return;
  }
  renderAnnotationState({
    user: state.user,
    state: {
      is_complete: false,
      current_item: state.currentItem,
      overall_progress: {
        completed_items: state.currentProgress.completed_overall,
        total_items: state.currentProgress.total_overall,
      },
    },
  });
});

refreshAdminButton.addEventListener("click", async () => {
  hideError(adminError);
  try {
    const payload = await fetchJson("/api/admin/dashboard");
    renderAdminDashboard(payload);
  } catch (error) {
    showError(adminError, error.message);
  }
});

bulkDownloadButton.addEventListener("click", async () => {
  hideError(adminError);
  try {
    const response = await fetch("/api/admin/download/raw-all", {
      headers: {
        Authorization: `Bearer ${state.sessionToken}`,
      },
    });
    if (!response.ok) {
      throw new Error("Bulk download failed.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "vocalgrad_raw_annotation_files.zip";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    showError(adminError, error.message);
  }
});

adminSummaryMode.addEventListener("change", async () => {
  hideError(adminError);
  try {
    const payload = await fetchJson("/api/admin/dashboard");
    renderAdminDashboard(payload);
  } catch (error) {
    showError(adminError, error.message);
  }
});

adminSummaryValue.addEventListener("change", async () => {
  hideError(adminError);
  try {
    const payload = await fetchJson("/api/admin/dashboard");
    renderAdminDashboard(payload);
  } catch (error) {
    showError(adminError, error.message);
  }
});

resetUserButton.addEventListener("click", async () => {
  hideError(adminError);
  const userId = resetUserSelect.value;
  if (!userId) {
    showError(adminError, "Select a user to reset.");
    return;
  }
  const confirmed = window.confirm(
    `Reset all saved annotation data for ${userId}? This cannot be undone.`,
  );
  if (!confirmed) {
    return;
  }
  try {
    const payload = await fetchJson("/api/admin/reset-user", {
      method: "POST",
      body: JSON.stringify({
        user_id: userId,
      }),
    });
    renderAdminDashboard(payload.dashboard);
  } catch (error) {
    showError(adminError, error.message);
  }
});

loadSessionState();
