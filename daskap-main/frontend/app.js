const API_URL = "http://localhost:8000/chat";

const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const chatMessages = document.getElementById("chatMessages");
const newChatButton = document.getElementById("newChatButton");
const messageTemplate = document.getElementById("messageTemplate");

const autoResize = () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 180)}px`;
};

const appendMessage = (role, text) => {
  const element = messageTemplate.content.firstElementChild.cloneNode(true);
  element.classList.add(`message--${role}`);
  element.querySelector(".message__avatar").textContent = role === "user" ? "You" : "D";
  element.querySelector(".message__content").textContent = text;
  chatMessages.appendChild(element);
  chatMessages.scrollTop = chatMessages.scrollHeight;
};

const setSendingState = (sending) => {
  sendButton.disabled = sending;
  sendButton.textContent = sending ? "Thinking..." : "Send";
};

const fetchBotReply = async (message) => {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }

  const payload = await response.json();
  return payload.reply ?? payload.response ?? "I received your message, but no reply field was returned.";
};

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();

  if (!text) {
    return;
  }

  appendMessage("user", text);
  messageInput.value = "";
  autoResize();
  setSendingState(true);

  try {
    const botReply = await fetchBotReply(text);
    appendMessage("assistant", botReply);
  } catch (error) {
    appendMessage(
      "assistant",
      "I couldn't reach the backend. Update API_URL in app.js and ensure your bot server allows CORS."
    );
    console.error(error);
  } finally {
    setSendingState(false);
    messageInput.focus();
  }
});

newChatButton.addEventListener("click", () => {
  chatMessages.innerHTML = "";
  appendMessage(
    "assistant",
    "New chat started. Ask me to summarize feedback, detect emotions, or draft action items."
  );
});

messageInput.addEventListener("input", autoResize);

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});
