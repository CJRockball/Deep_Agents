// IntelliFinQ Chat JavaScript
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("user-input");
  const windowDiv = document.getElementById("chat-window");
  const userId = "user123";  // Could be dynamic

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    // Display user message
    appendMessage("user", question);
    input.value = "";

    // Call backend API
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ user_id: userId, query: question })
      });
      const data = await resp.json();
      appendMessage("agent", data.response);
    } catch (err) {
      appendMessage("agent", "Error: " + err.message);
    }
  });

  function appendMessage(role, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message " + role;
    msgDiv.innerText = text;
    windowDiv.appendChild(msgDiv);
    windowDiv.scrollTop = windowDiv.scrollHeight;
  }
});
