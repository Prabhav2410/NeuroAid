document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const messagesContainer = document.getElementById("messages");
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");
  const newChatButton = document.getElementById("new-chat");
  const clearHistoryButton = document.getElementById("clear-history");
  const toggleThemeButton = document.getElementById("toggle-theme");
  const chatHistoryContainer = document.getElementById("chat-history");
  const currentChatTitle = document.getElementById("current-chat-title");
  const exportChatButton = document.getElementById("export-chat");
  const regenerateResponseButton = document.getElementById(
    "regenerate-response"
  );
  const stopResponseButton = document.getElementById("stop-response");
  const suggestionChips = document.querySelectorAll(".suggestion-chip");
  const fileUploadButton = document.getElementById("file-upload-button");
  const fileUploadInput = document.getElementById("file-upload");

  // OpenRouter API Key and Configuration
  const API_KEY = "GEMINI_API_KEY";

  // State variables
  let currentChatId = null;
  let isTyping = false;
  let chatHistory = JSON.parse(localStorage.getItem("chatHistory")) || {};
  let currentTheme = localStorage.getItem("theme") || "light";
  let typingSpeed = 2; // reduced for faster letter-by-letter output
  let letterTimeout = null; // separate timeout for letter typing
  let pendingFile = null; // Store selected file until send
  let stopGeneration = false; // Flag to stop the typing effect

  // Helper function to get stable rendering:
  function getStableRendering(text) {
    // Split text on triple backticks.
    let parts = text.split("```");
    if (parts.length % 2 === 1) {
      // All code blocks are closed.
      return marked.parse(text);
    } else {
      // Code block is open; force rendering as a code block by artificially closing it.
      let closedPart = parts.slice(0, parts.length - 1).join("```");
      let openPart = parts[parts.length - 1];
      return (
        marked.parse(closedPart) + marked.parse("```" + openPart + "\n```")
      );
    }
  }

  // Initialize application
  init();

  // Function to initialize the application
  function init() {
    // Apply theme
    if (currentTheme === "dark") {
      document.body.classList.add("dark-mode");
      toggleThemeButton.innerHTML =
        '<i class="fas fa-sun"></i><span>Light Mode</span>';
    }

    // Set up textarea auto-resize
    userInput.addEventListener("input", autoResizeTextarea);

    // Set up event listeners
    sendButton.addEventListener("click", handleSendMessage);
    userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    newChatButton.addEventListener("click", createNewChat);
    clearHistoryButton.addEventListener("click", clearAllHistory);
    toggleThemeButton.addEventListener("click", toggleTheme);
    exportChatButton.addEventListener("click", exportCurrentChat);
    regenerateResponseButton.addEventListener("click", regenerateLastResponse);
    stopResponseButton.addEventListener("click", () => {
      stopGeneration = true;
      clearTimeout(letterTimeout);
      stopResponseButton.style.display = "none";
      regenerateResponseButton.style.display = "inline-block";
    });

    fileUploadButton.addEventListener("click", () => {
      fileUploadInput.click();
    });
    // Store pending file and show preview on selection
    fileUploadInput.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) {
        pendingFile = file;
        displayPendingFilePreview(file);
      }
    });

    suggestionChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        userInput.value = chip.textContent;
        handleSendMessage();
      });
    });

    // Load chat history
    updateChatHistorySidebar();

    // Create new chat if none exists
    if (Object.keys(chatHistory).length === 0) {
      createNewChat();
    } else {
      // Load most recent chat
      const mostRecentChatId = Object.keys(chatHistory).sort((a, b) => {
        return chatHistory[b].timestamp - chatHistory[a].timestamp;
      })[0];

      loadChat(mostRecentChatId);
    }

    // Global click listener to close any open options menus
    window.addEventListener("click", () => {
      document.querySelectorAll(".chat-options-menu").forEach((menu) => {
        menu.style.display = "none";
      });
    });
  }

  // NEW: Display pending file preview immediately after selection
  function displayPendingFilePreview(file) {
    const previewContainer = document.getElementById("pending-file-preview");
    const reader = new FileReader();
    reader.onload = function (e) {
      let previewHTML = "";
      if (file.type.startsWith("image/")) {
        previewHTML = `<img src="${e.target.result}" alt="${file.name}" style="max-width: 100px; max-height: 100px;"/>`;
      } else if (
        file.type.startsWith("text/") ||
        file.type === "application/json"
      ) {
        let content = e.target.result;
        if (content.length > 200) {
          content = content.substring(0, 200) + "...";
        }
        previewHTML = `<pre style="white-space: pre-wrap; font-size: 12px;">${escapeHtml(
          content
        )}</pre>`;
      } else {
        previewHTML = `<div style="font-size: 12px;">${file.name}</div>`;
      }
      previewContainer.innerHTML = previewHTML;
      previewContainer.style.display = "block";
    };

    if (file.type.startsWith("image/")) {
      reader.readAsDataURL(file);
    } else {
      reader.readAsText(file);
    }
  }

  // NEW: Process pending file when sending message
  function processPendingFile() {
    return new Promise((resolve, reject) => {
      const file = pendingFile;
      if (!file) {
        resolve();
        return;
      }
      const reader = new FileReader();
      reader.onload = function (e) {
        let previewHTML = "";
        if (file.type.startsWith("image/")) {
          previewHTML = `<img src="${e.target.result}" alt="${file.name}" style="max-width:100%;"/>`;
        } else if (
          file.type.startsWith("text/") ||
          file.type === "application/json"
        ) {
          previewHTML = `<pre style="white-space: pre-wrap;">${escapeHtml(
            e.target.result
          )}</pre>`;
        } else {
          previewHTML = `<div>Uploaded file: ${file.name}</div>`;
        }
        // Append file preview as a user message
        addFileMessageToUI("user", previewHTML);
        // Save file data in chat history
        if (!chatHistory[currentChatId]) {
          createNewChat();
        }
        chatHistory[currentChatId].messages.push({
          role: "user",
          content: previewHTML,
          file: {
            name: file.name,
            type: file.type,
            content: e.target.result
          }
        });
        // Clear pending file preview
        const previewContainer = document.getElementById(
          "pending-file-preview"
        );
        previewContainer.style.display = "none";
        previewContainer.innerHTML = "";
        pendingFile = null;
        resolve();
      };

      if (file.type.startsWith("image/")) {
        reader.readAsDataURL(file);
      } else {
        reader.readAsText(file);
      }
    });
  }

  // Helper to escape HTML (for text files)
  function escapeHtml(text) {
    var map = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    };
    return text.replace(/[&<>"']/g, function (m) {
      return map[m];
    });
  }

  // Function to add file message (with HTML preview) to UI
  function addFileMessageToUI(sender, htmlContent) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;
    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.innerHTML = htmlContent;
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Function to handle sending a message
  async function handleSendMessage() {
    if (isTyping) {
      alert("Please wait until the current response is completed.");
      return;
    }
    const message = userInput.value.trim();
    // Only proceed if there is text or a pending file
    if (!message && !pendingFile) return;

    // Clear input and reset height
    userInput.value = "";
    userInput.style.height = "auto";

    // If there's text, add it as a user message
    if (message) {
      addMessageToUI("user", message);

      if (!chatHistory[currentChatId]) {
        createNewChat();
      }

      chatHistory[currentChatId].messages.push({
        role: "user",
        content: message
      });

      if (chatHistory[currentChatId].messages.length === 1) {
        const title =
          message.split(" ").slice(0, 4).join(" ") +
          (message.split(" ").length > 4 ? "..." : "");
        chatHistory[currentChatId].title = title;
        currentChatTitle.textContent = title;
        updateChatHistorySidebar();
      }
    }

    // If a file was selected, process it now
    if (pendingFile) {
      await processPendingFile();
    }

    saveChatHistory();

    try {
      showTypingIndicator();
      // Stream the AI response and update UI letter-by-letter
      const response = await getAIResponse(currentChatId);
      // The UI is updated during streaming. Save the final text.
      chatHistory[currentChatId].messages.push({
        role: "assistant",
        content: response
      });

      saveChatHistory();
    } catch (error) {
      removeTypingIndicator();
      addMessageToUIWithTypingEffect(
        "ai",
        `Sorry, I encountered an error: ${error.message}`
      );
    }
  }

  // UPDATED: Function to get AI response from OpenRouter API with streaming and letter-by-letter effect.
  // While streaming, we update the message content using getStableRendering so that if a code block is open,
  // it is rendered as a code block immediately.
  // Function to get AI response from your Flask backend
  // Function to get AI response from Flask backend (non-streaming)
async function getAIResponse(chatId) {
  isTyping = true;
  stopGeneration = false;
  regenerateResponseButton.style.display = "none";
  stopResponseButton.style.display = "inline-block";

  try {
    const lastMessage =
      chatHistory[chatId].messages[chatHistory[chatId].messages.length - 1];

    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: lastMessage.content })
    });

    if (!response.ok) {
      throw new Error("Failed to get response from server");
    }

    const data = await response.json();
    const reply = data.reply || "(no reply received)";

    // Show AI reply in UI
    addMessageToUI("ai", reply);


    // Save reply in history
    chatHistory[chatId].messages.push({
      role: "assistant",
      content: reply
    });
    saveChatHistory();

    isTyping = false;
    removeTypingIndicator();
    stopResponseButton.style.display = "none";
    regenerateResponseButton.style.display = "inline-block";

    return reply;
  } catch (error) {
    isTyping = false;
    removeTypingIndicator();
    console.error("API Error:", error);
    addMessageToUIWithTypingEffect(
      "ai",
      `Sorry, I encountered an error: ${error.message}`
    );
    throw error;
  }
}

  // Function to add message to UI with typing effect (used for non-streaming messages)
  function addMessageToUIWithTypingEffect(sender, content) {
    removeTypingIndicator();

    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";

    if (sender === "user") {
      messageContent.textContent = content;
      messageDiv.appendChild(messageContent);
      messagesContainer.appendChild(messageDiv);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      return;
    }

    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);

    const processedContent = processMarkdownContent(content);

    // When AI response starts, reset the stop flag and show the stop button
    if (sender === "ai") {
      stopGeneration = false;
      stopResponseButton.style.display = "inline-block";
    }

    startTypingEffect(messageContent, processedContent, 0);
  }

  // Function to add user message to UI immediately
  function addMessageToUI(sender, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.textContent = content;

    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Function to process markdown and identify code blocks
  function processMarkdownContent(content) {
    const segments = [];
    let currentPos = 0;
    const codeBlockRegex = /```([\w]*)\n([\s\S]*?)\n```/g;

    let match;
    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > currentPos) {
        segments.push({
          type: "text",
          content: content.substring(currentPos, match.index)
        });
      }

      segments.push({
        type: "code",
        language: match[1] || "plaintext",
        content: match[2]
      });

      currentPos = match.index + match[0].length;
    }

    if (currentPos < content.length) {
      segments.push({
        type: "text",
        content: content.substring(currentPos)
      });
    }

    return segments;
  }

  // Function to start typing effect (used by non-streaming messages)
  function startTypingEffect(messageContent, segments, segmentIndex) {
    if (segmentIndex >= segments.length) {
      isTyping = false;
      stopResponseButton.style.display = "none";
      return;
    }

    const segment = segments[segmentIndex];

    if (segment.type === "code") {
      const preElement = document.createElement("pre");
      const codeElement = document.createElement("code");

      if (segment.language) {
        codeElement.className = `language-${segment.language}`;
      }
      codeElement.classList.add("hljs");

      const copyButtonContainer = document.createElement("div");
      copyButtonContainer.className = "code-copy-container";

      const copyButton = document.createElement("button");
      copyButton.className = "code-copy-button";
      copyButton.innerHTML = '<i class="fas fa-copy"></i>';
      copyButton.title = "Copy code";

      copyButton.addEventListener("click", () => {
        navigator.clipboard.writeText(segment.content).then(() => {
          copyButton.innerHTML = '<i class="fas fa-check"></i>';
          copyButton.classList.add("copied");
          setTimeout(() => {
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            copyButton.classList.remove("copied");
          }, 2000);
        });
      });

      copyButtonContainer.appendChild(copyButton);
      preElement.appendChild(copyButtonContainer);

      preElement.appendChild(codeElement);
      messageContent.appendChild(preElement);

      typeCodeContent(codeElement, segment.content, 0, () => {
        hljs.highlightElement(codeElement);
        startTypingEffect(messageContent, segments, segmentIndex + 1);
      });
    } else {
      const textDiv = document.createElement("div");
      messageContent.appendChild(textDiv);
      typeTextContent(textDiv, segment.content, 0, () => {
        startTypingEffect(messageContent, segments, segmentIndex + 1);
      });
    }
  }

  // Function to type code content letter by letter
  function typeCodeContent(element, content, index, callback) {
    if (stopGeneration) {
      callback();
      return;
    }
    if (index < content.length) {
      element.textContent += content[index];
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      letterTimeout = setTimeout(() => {
        typeCodeContent(element, content, index + 1, callback);
      }, typingSpeed);
    } else {
      callback();
    }
  }

  // Function to type text content letter by letter
  function typeTextContent(element, content, index, callback) {
    if (stopGeneration) {
      callback();
      return;
    }
    if (index < content.length) {
      let currentText = content.substring(0, index + 1);
      element.innerHTML = marked.parse(currentText);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      letterTimeout = setTimeout(() => {
        typeTextContent(element, content, index + 1, callback);
      }, typingSpeed);
    } else {
      callback();
    }
  }

  // Function to show typing indicator
  function showTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "typing-indicator";
    typingDiv.id = "typing-indicator";

    for (let i = 0; i < 3; i++) {
      const dot = document.createElement("div");
      dot.className = "typing-dot";
      typingDiv.appendChild(dot);
    }

    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Function to remove typing indicator
  function removeTypingIndicator() {
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }

  // Function to create a new chat
  function createNewChat() {
    const chatId = "chat_" + Date.now();

    chatHistory[chatId] = {
      id: chatId,
      title: "New Conversation",
      timestamp: Date.now(),
      messages: []
    };

    currentChatId = chatId;
    currentChatTitle.textContent = "New Conversation";

    messagesContainer.innerHTML = `
            <div class="intro-message">
                <h1>Welcome to NeoChat AI</h1>
                <p>Ask me anything. I'm powered by gemini.</p>
                <div class="suggestion-chips">
                    <button class="suggestion-chip">Help me find my disease</button>
                    <button class="suggestion-chip">Explain my treatment options</button>
                    <button class="suggestion-chip">Explain my symptoms</button>
                    <button class="suggestion-chip">Help me learn about my condition</button>
                </div>
            </div>
          `;

    document.querySelectorAll(".suggestion-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        userInput.value = chip.textContent;
        handleSendMessage();
      });
    });

    saveChatHistory();
    updateChatHistorySidebar();
    // Clear pending file and its preview when a new chat is created
    pendingFile = null;
    const previewContainer = document.getElementById("pending-file-preview");
    if (previewContainer) {
      previewContainer.innerHTML = "";
      previewContainer.style.display = "none";
    }
  }

  // Function to load a chat
  function loadChat(chatId) {
    if (!chatHistory[chatId]) return;

    currentChatId = chatId;
    currentChatTitle.textContent = chatHistory[chatId].title;

    messagesContainer.innerHTML = "";

    chatHistory[chatId].messages.forEach((message) => {
      // If the user message has a file property, render using innerHTML
      if (message.role === "user") {
        if (message.file) {
          addFileMessageToUI("user", message.content);
        } else {
          addMessageToUI("user", message.content);
        }
      } else {
        addFormattedMessageToUI("ai", message.content);
      }
    });

    updateActiveChatInSidebar();
  }

  // Function to add formatted message to UI without typing effect
  function addFormattedMessageToUI(sender, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";

    const processedContent = processMarkdownContent(content);

    processedContent.forEach((segment) => {
      if (segment.type === "code") {
        const preElement = document.createElement("pre");
        const codeElement = document.createElement("code");

        if (segment.language) {
          codeElement.className = `language-${segment.language}`;
        }
        codeElement.classList.add("hljs");

        const copyButtonContainer = document.createElement("div");
        copyButtonContainer.className = "code-copy-container";

        const copyButton = document.createElement("button");
        copyButton.className = "code-copy-button";
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.title = "Copy code";

        copyButton.addEventListener("click", () => {
          navigator.clipboard.writeText(segment.content).then(() => {
            copyButton.innerHTML = '<i class="fas fa-check"></i>';
            copyButton.classList.add("copied");
            setTimeout(() => {
              copyButton.innerHTML = '<i class="fas fa-copy"></i>';
              copyButton.classList.remove("copied");
            }, 2000);
          });
        });

        copyButtonContainer.appendChild(copyButton);
        preElement.appendChild(copyButtonContainer);

        codeElement.textContent = segment.content;
        preElement.appendChild(codeElement);
        messageContent.appendChild(preElement);

        hljs.highlightElement(codeElement);
      } else {
        const textDiv = document.createElement("div");
        textDiv.innerHTML = marked.parse(segment.content);

        while (textDiv.firstChild) {
          messageContent.appendChild(textDiv.firstChild);
        }
      }
    });

    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Function to save chat history to localStorage
  function saveChatHistory() {
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    updateChatHistorySidebar();
  }

  // Function to update chat history sidebar with options (delete/rename)
  function updateChatHistorySidebar() {
    chatHistoryContainer.innerHTML = "";

    const sortedChatIds = Object.keys(chatHistory).sort((a, b) => {
      return chatHistory[b].timestamp - chatHistory[a].timestamp;
    });

    sortedChatIds.forEach((chatId) => {
      const chat = chatHistory[chatId];
      const chatItem = document.createElement("div");
      chatItem.className = `chat-history-item ${
        chatId === currentChatId ? "active" : ""
      }`;
      chatItem.dataset.chatId = chatId;

      // Create icon element
      const icon = document.createElement("i");
      icon.className = "fas fa-comment";
      // Create span for chat title using textContent for safety
      const titleSpan = document.createElement("span");
      titleSpan.textContent = chat.title;

      chatItem.appendChild(icon);
      chatItem.appendChild(titleSpan);

      // When clicking the chat item (outside the options button) load the chat
      chatItem.addEventListener("click", () => {
        loadChat(chatId);
      });

      // Create the options button (three dots)
      const optionsButton = document.createElement("button");
      optionsButton.className = "chat-options-button";
      optionsButton.innerHTML = '<i class="fas fa-ellipsis-v"></i>';
      // Stop propagation so clicking this does not trigger the parent click event
      optionsButton.addEventListener("click", (e) => {
        e.stopPropagation();
        // Toggle the options menu
        optionsMenu.style.display =
          optionsMenu.style.display === "none" ? "block" : "none";
      });

      // Create the options menu (hidden by default)
      const optionsMenu = document.createElement("div");
      optionsMenu.className = "chat-options-menu";
      optionsMenu.style.display = "none";
      optionsMenu.innerHTML = `
              <div class="chat-options-item delete-chat">Delete</div>
              <div class="chat-options-item rename-chat">Rename</div>
            `;

      // Handle deletion of a chat
      optionsMenu
        .querySelector(".delete-chat")
        .addEventListener("click", (e) => {
          e.stopPropagation();
          if (confirm("Are you sure you want to delete this chat?")) {
            delete chatHistory[chatId];
            if (currentChatId === chatId) {
              createNewChat();
            }
            saveChatHistory();
            updateChatHistorySidebar();
          }
        });

      // Handle renaming a chat
      optionsMenu
        .querySelector(".rename-chat")
        .addEventListener("click", (e) => {
          e.stopPropagation();
          const newName = prompt("Enter new name for this chat:", chat.title);
          if (newName) {
            chat.title = newName;
            saveChatHistory();
            updateChatHistorySidebar();
            if (currentChatId === chatId) {
              currentChatTitle.textContent = newName;
            }
          }
        });

      // Append the options button and menu to the chat item
      chatItem.appendChild(optionsButton);
      chatItem.appendChild(optionsMenu);

      // Append the chat item to the history container
      chatHistoryContainer.appendChild(chatItem);
    });
  }

  // Function to update active chat in sidebar
  function updateActiveChatInSidebar() {
    document.querySelectorAll(".chat-history-item").forEach((item) => {
      item.classList.remove("active");
      if (item.dataset.chatId === currentChatId) {
        item.classList.add("active");
      }
    });
  }

  // Function to clear all chat history
  function clearAllHistory() {
    if (
      confirm(
        "Are you sure you want to clear all chat history? This cannot be undone."
      )
    ) {
      chatHistory = {};
      localStorage.removeItem("chatHistory");
      createNewChat();
    }
  }

  // Function to toggle theme
  function toggleTheme() {
    if (currentTheme === "light") {
      document.body.classList.add("dark-mode");
      currentTheme = "dark";
      toggleThemeButton.innerHTML =
        '<i class="fas fa-sun"></i><span>Light Mode</span>';
    } else {
      document.body.classList.remove("dark-mode");
      currentTheme = "light";
      toggleThemeButton.innerHTML =
        '<i class="fas fa-moon"></i><span>Dark Mode</span>';
    }

    localStorage.setItem("theme", currentTheme);
  }

  // Function to export current chat
  function exportCurrentChat() {
    if (!chatHistory[currentChatId]) return;

    const chat = chatHistory[currentChatId];
    let exportText = `# ${chat.title}\n\n`;

    chat.messages.forEach((message) => {
      const role = message.role === "user" ? "You" : "NeoChat AI";
      exportText += `## ${role}:\n${message.content}\n\n`;
    });

    const blob = new Blob([exportText], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `${chat.title.replace(/[^\w\s]/gi, "")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Function to regenerate the last AI response
  function regenerateLastResponse() {
    if (
      chatHistory[currentChatId] &&
      chatHistory[currentChatId].messages.length > 0
    ) {
      const lastMessage =
        chatHistory[currentChatId].messages[
          chatHistory[currentChatId].messages.length - 1
        ];
      if (lastMessage.role === "assistant") {
        chatHistory[currentChatId].messages.pop();
        const messageElements = document.querySelectorAll(".message.ai");
        if (messageElements.length > 0) {
          messageElements[messageElements.length - 1].remove();
        }
        saveChatHistory();
        showTypingIndicator();
        getAIResponse(currentChatId)
          .then((response) => {
            chatHistory[currentChatId].messages.push({
              role: "assistant",
              content: response
            });
            saveChatHistory();
          })
          .catch((error) => {
            removeTypingIndicator();
            addMessageToUIWithTypingEffect(
              "ai",
              `Sorry, I encountered an error: ${error.message}`
            );
          });
      }
    }
  }

  // Function to auto-resize textarea
  function autoResizeTextarea() {
    userInput.style.height = "auto";
    userInput.style.height = userInput.scrollHeight + "px";
  }
});
