// DOM Elements
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-message');
const chatMessages = document.getElementById('chat-messages');
const websiteUrlInput = document.getElementById('website-url');
const startScrapeButton = document.getElementById('start-scrape');
const resetDbButton = document.getElementById('reset-db');
const scrapingStatus = document.getElementById('scraping-status');
const statusText = document.getElementById('status-text');
const docCount = document.getElementById('doc-count');
const modelName = document.getElementById('model-name');
const ollamaStatus = document.getElementById('ollama-status');
const copyEmbedButton = document.getElementById('copy-embed');
const embedUrl = document.getElementById('embed-url');
const modeFlashBtn = document.getElementById('mode-flash');
const modeDeepAnalysisBtn = document.getElementById('mode-deep-analysis');
const modeDeepBtn = document.getElementById('mode-deep');
const modeDeepseekBtn = document.getElementById('mode-deepseek');
// History related elements are now handled directly in the HTML
const stopProcessBtn = document.getElementById('stop-process');

// Set the embed URL based on current location
embedUrl.textContent = `${window.location.origin}/embed/chatbot.js`;

// Check system status on page load
document.addEventListener('DOMContentLoaded', () => {
    checkSystemStatus();
    
    // Add tooltip functionality for mode buttons
    document.querySelectorAll('.mode-button[title]').forEach(button => {
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = button.getAttribute('title');
        button.appendChild(tooltip);
        
        button.addEventListener('mouseenter', () => {
            tooltip.style.opacity = '1';
            tooltip.style.visibility = 'visible';
        });
        
        button.addEventListener('mouseleave', () => {
            tooltip.style.opacity = '0';
            tooltip.style.visibility = 'hidden';
        });
    });
    
    // Focus on URL input when page loads
    websiteUrlInput.focus();
});

// --- Mode selection logic ---
let chatMode = 'flash';

const modeBtns = [modeFlashBtn, modeDeepAnalysisBtn, modeDeepBtn, modeDeepseekBtn];

if (modeBtns.every(btn => btn)) {
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Get the mode from the button's id
            const mode = btn.id.replace('mode-', '');
            chatMode = mode;
            
            // Update selected state
            modeBtns.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            
            // Provide feedback to user
            addSystemMessage(`Switched to ${btn.textContent.trim()} mode.`);
        });
    });
}

// Create and add AI provider select
// Removed AI provider select as AI Extract mode is removed

// Event Listeners
startScrapeButton.addEventListener('click', startScraping);
resetDbButton.addEventListener('click', resetDatabase);
sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Copy embed code
copyEmbedButton.addEventListener('click', copyEmbedCode);

// Stop process button
stopProcessBtn.addEventListener('click', () => {
    if (currentController) {
        currentController.abort();
        addSystemMessage('Process stopped by user.');
    }
    showStopButton(false);
    sendButton.disabled = false;
});

// Periodically check status if scraping is in progress
let statusCheckInterval = null;

/**
 * Check system status
 */
function checkSystemStatus() {
    fetch('/api/status')
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(`Server responded with ${response.status}: ${errorData.error}\nDetails: ${errorData.details || 'No details provided'}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'ok') {
                docCount.textContent = data.database.document_count;
                modelName.textContent = data.model_name;
                ollamaStatus.textContent = 'Connected';
                ollamaStatus.style.color = 'var(--success-color)';
                
                // Enable chat if documents exist
                if (data.database.document_count > 0) {
                    enableChat();
                }
            } else {
                ollamaStatus.textContent = 'Disconnected';
                ollamaStatus.style.color = 'var(--danger-color)';
                addSystemMessage('Error connecting to Ollama. Make sure Ollama is running and the model is downloaded.');
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            ollamaStatus.textContent = 'Error';
            ollamaStatus.style.color = 'var(--danger-color)';
            addSystemMessage(`Error checking status: ${error.message}`);
        });
}

/**
 * Start scraping process
 */
function startScraping() {
    const url = websiteUrlInput.value.trim();
    
    if (!url) {
        addSystemMessage('Please enter a valid website URL.');
        websiteUrlInput.focus();
        return;
    }
    
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        addSystemMessage('URL must start with http:// or https://');
        websiteUrlInput.focus();
        return;
    }
    
    // Disable input during scraping
    websiteUrlInput.disabled = true;
    startScrapeButton.disabled = true;
    
    // Show scraping status
    scrapingStatus.classList.remove('hidden', 'success', 'error');
    statusText.textContent = `Scraping in progress for ${url}...`;
    
    // Start scraping
    fetch('/api/scrape', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(errorData => {
                throw new Error(`Server responded with ${response.status}: ${errorData.error}\nDetails: ${errorData.details || 'No details provided'}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            showScrapeError(data.error);
            return;
        }
        
        // Start checking status periodically
        statusCheckInterval = setInterval(checkScrapeStatus, 2000);
        addSystemMessage(`Started scraping ${url}. This may take a few minutes depending on the size of the website.`);
    })
    .catch(error => {
        console.error('Error starting scrape:', error);
        showScrapeError('Failed to start scraping. Please try again.');
        addSystemMessage(`Error starting scrape: ${error.message}`);
    });
}

/**
 * Check scraping status
 */
function checkScrapeStatus() {
    fetch('/api/scrape/status')
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(`Server responded with ${response.status}: ${errorData.error}\nDetails: ${errorData.details || 'No details provided'}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                clearInterval(statusCheckInterval);
                showScrapeError(data.error);
                return;
            }
            
            if (!data.is_scraping) {
                // Scraping finished
                clearInterval(statusCheckInterval);
                scrapingStatus.classList.add('success');
                statusText.textContent = `Scraping complete! Processed ${data.documents_processed} documents.`;
                
                // Re-enable inputs
                websiteUrlInput.disabled = false;
                startScrapeButton.disabled = false;
                
                // Check system status to update doc count
                checkSystemStatus();
                
                // Enable chat
                enableChat();
                
                // Add success message
                addSystemMessage(`Scraping complete! Processed ${data.documents_processed} documents from ${data.current_url}. You can now ask questions about the website content.`);
                
                // Remove status after a delay
                setTimeout(() => {
                    scrapingStatus.classList.remove('success');
                    scrapingStatus.classList.add('hidden');
                }, 5000);
            } else {
                // Update status text
                statusText.textContent = `Scraping in progress for ${data.current_url}... (${data.documents_processed} documents so far)`;
            }
        })
        .catch(error => {
            console.error('Error checking scrape status:', error);
            addSystemMessage(`Error checking scrape status: ${error.message}`);
        });
}

/**
 * Show scraping error
 */
function showScrapeError(errorMessage) {
    scrapingStatus.classList.add('error');
    statusText.textContent = `Error: ${errorMessage}`;
    
    // Re-enable inputs
    websiteUrlInput.disabled = false;
    startScrapeButton.disabled = false;
    
    // Add error message
    addSystemMessage(`Scraping failed: ${errorMessage}`);
    
    // Remove status after a delay
    setTimeout(() => {
        scrapingStatus.classList.remove('error');
        scrapingStatus.classList.add('hidden');
    }, 5000);
}

/**
 * Reset database
 */
function resetDatabase() {
    if (!confirm('Are you sure you want to reset the database? This will delete all scraped content.')) {
        return;
    }
    
    fetch('/api/reset', {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(errorData => {
                throw new Error(`Server responded with ${response.status}: ${errorData.error}\nDetails: ${errorData.details || 'No details provided'}`);
            });
        }
        return response.json();
    })
    .then(data => {
        addSystemMessage('Database reset successfully. You need to scrape a website again before asking questions.');
        disableChat();
        checkSystemStatus();
    })
    .catch(error => {
        console.error('Error resetting database:', error);
        addSystemMessage(`Error resetting database: ${error.message}`);
    });
}

/**
 * Enable chat interface
 */
function enableChat() {
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
    
    // Add a subtle animation to indicate chat is enabled
    chatInput.classList.add('enabled');
    setTimeout(() => {
        chatInput.classList.remove('enabled');
    }, 1000);
}

/**
 * Disable chat interface
 */
function disableChat() {
    chatInput.disabled = true;
    sendButton.disabled = true;
}

/**
 * Show/hide stop button
 */
let currentController = null;

function showStopButton(show = true) {
    if (show) {
        stopProcessBtn.style.display = '';
    } else {
        stopProcessBtn.style.display = 'none';
    }
}

/**
 * Send a message
 */
function sendMessage() {
    if (chatInput.disabled) return;
    
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addUserMessage(message);
    chatInput.value = '';
    
    // Show stop button
    showStopButton();
    
    // Add typing indicator
    const botTypingIndicator = addBotTypingIndicator();
    
    // Different API endpoints depending on mode
    let apiEndpoint;
    let requestData;
    
    if (chatMode === 'deep-analysis') {
        // Deep analysis mode - uses different endpoint
        apiEndpoint = '/api/ollama_deepseek';
        requestData = {
            prompt: message
        };
    } else if (chatMode === 'deepseek') {
        // DeepSeek code mode
        apiEndpoint = '/api/query';
        requestData = {
            query: message,
            mode: 'deepseek',
            temperature: 0.4,
            max_tokens: 1024
        };
    } else {
        // Standard RAG query (flash, deep)
        apiEndpoint = '/api/query';
        requestData = {
            query: message,
            mode: chatMode
        };
    }
    
    // Disable input while waiting for response
    chatInput.disabled = true;
    sendButton.disabled = true;
    
    // Create an AbortController for the fetch request
    currentController = new AbortController();
    
    // Make request to server
    fetch(apiEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData),
        signal: currentController.signal
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(errorData => {
                throw new Error(`Server responded with ${response.status}: ${errorData.error}\nDetails: ${errorData.details || 'No details provided'}`);
            });
        }
        return response.json();
    })
    .then(data => {
        // Remove typing indicator
        chatMessages.removeChild(botTypingIndicator);
        
        // Process response based on mode and endpoint
        if (chatMode === 'deep-analysis' && data.result) {
            // For deep analysis, the response is in data.result
            addBotMessage(data.result);
            saveToHistory(message, data.result);
        } else if (data.answer) {
            // For standard modes, the response is in data.answer
            addBotMessage(data.answer);
            
            // Add sources if available
            if (data.contexts && data.contexts.length > 0) {
                addSourcesInfo(data.contexts);
            }
            
            saveToHistory(message, data.answer);
        } else if (data.error) {
            addSystemMessage(`Error: ${data.error}`);
        } else {
            addSystemMessage('Received an empty response from the server.');
        }
        
        // Re-enable input
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
        
        // Hide stop button
        showStopButton(false);
    })
    .catch(error => {
        if (error.name === 'AbortError') {
            console.log('Request was aborted');
        } else {
            console.error('Error:', error);
            chatMessages.removeChild(botTypingIndicator);
            showStopButton(false);
            addSystemMessage(`Error: ${error.message}`);
        }
    })
    .finally(() => {
        // Reset controller
        currentController = null;
        
        // Re-enable input
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    });
}

/**
 * Add a user message to the chat
 */
function addUserMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message user';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = message;
    
    messageElement.appendChild(messageContent);
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    scrollToBottom();
}

/**
 * Add a bot message to the chat
 */
function addBotMessage(message, contexts) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message bot';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Parse markdown
    messageContent.innerHTML = marked.parse(message);
    
    // Add syntax highlighting for code blocks
    messageContent.querySelectorAll('pre code').forEach(block => {
        if (window.hljs) {
            window.hljs.highlightElement(block);
        }
    });
    
    messageElement.appendChild(messageContent);
    
    // Add sources if available
    if (contexts && contexts.length > 0) {
        const sourcesElement = document.createElement('div');
        sourcesElement.className = 'sources';
        
        // Get unique sources
        const uniqueSources = new Map();
        contexts.forEach(context => {
            if (context.source && context.title) {
                uniqueSources.set(context.source, context.title);
            }
        });
        
        const sourcesList = Array.from(uniqueSources.entries())
            .map(([url, title]) => `<a href="${url}" target="_blank" rel="noopener noreferrer"><i class="fas fa-external-link-alt"></i> ${title}</a>`)
            .join(', ');
        
        sourcesElement.innerHTML = `<strong>Sources:</strong> ${sourcesList}`;
        messageElement.appendChild(sourcesElement);
    }
    
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    scrollToBottom();
}

/**
 * Add a system message to the chat
 */
function addSystemMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message system';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = `<i class="fas fa-info-circle"></i> ${message}`;
    
    messageElement.appendChild(messageContent);
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    scrollToBottom();
}

/**
 * Add a typing indicator
 */
function addBotTypingIndicator() {
    const messageElement = document.createElement('div');
    messageElement.className = 'message bot typing';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    
    messageElement.appendChild(messageContent);
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    scrollToBottom();
    
    return messageElement;
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Copy embed code
 */
function copyEmbedCode() {
    const embedCode = `<script src="${window.location.origin}/embed/chatbot.js"></script>`;
    navigator.clipboard.writeText(embedCode)
        .then(() => {
            const originalText = copyEmbedButton.textContent;
            copyEmbedButton.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => {
                copyEmbedButton.innerHTML = '<i class="fas fa-copy"></i> Copy';
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy: ', err);
            addSystemMessage('Failed to copy to clipboard. Please try again.');
        });
}

// Function stubs for backward compatibility - these are now handled in the HTML
function showChatHistory() {
    // This function is now empty as it's handled directly in the HTML
    console.log("History functionality moved to inline HTML script");
}

function downloadChatHistory() {
    // This function is now empty as it's handled directly in the HTML
    console.log("Download history functionality moved to inline HTML script");
}

/**
 * Handle fetch response
 */
function handleResponse(response) {
    if (!response.ok) {
        return response.json().then(errorData => {
            throw new Error(`Server responded with ${response.status}: ${errorData.error}\nDetails: ${errorData.details || 'No details provided'}`);
        });
    }
    return response.json();
}

/**
 * Add sources information to the last bot message
 */
function addSourcesInfo(contexts) {
    if (!contexts || contexts.length === 0) return;
    
    let sourcesHtml = '<div class="sources-container"><h4>Sources:</h4><ul>';
    contexts.forEach(context => {
        if (context.source) {
            sourcesHtml += `<li><a href="${context.source}" target="_blank">${context.title || context.source}</a></li>`;
        }
    });
    sourcesHtml += '</ul></div>';
    
    // Append sources to the last bot message
    const lastBotMessage = document.querySelector('.chat-message.bot:last-child .message-content');
    if (lastBotMessage) {
        lastBotMessage.innerHTML += sourcesHtml;
    }
}

/**
 * Save message to history
 */
function saveToHistory(userMessage, botResponse) {
    // Save to chat history via API
    fetch('/api/chat_history', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            user_message: userMessage,
            bot_response: botResponse
        })
    })
    .then(response => {
        if (!response.ok) {
            console.error('Failed to save chat history');
        }
    })
    .catch(error => {
        console.error('Error saving chat history:', error);
    });
}

// Add CSS for tooltip
const style = document.createElement('style');
style.textContent = `
.tooltip {
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background-color: var(--dark-color);
    color: white;
    padding: 0.5rem;
    border-radius: var(--border-radius-sm);
    font-size: 0.75rem;
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.3s, visibility 0.3s;
    z-index: 100;
    margin-bottom: 5px;
    box-shadow: var(--shadow);
}

.tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-width: 5px;
    border-style: solid;
    border-color: var(--dark-color) transparent transparent transparent;
}

.mode-button {
    position: relative;
}

.input-with-icon {
    position: relative;
}

.input-with-icon i {
    position: absolute;
    left: 10px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--gray-500);
}

.input-with-icon input {
    padding-left: 35px !important;
}

.input-group {
    display: flex;
    gap: 10px;
}

.chat-input-container .input-group {
    margin-top: 10px;
}

.button-group {
    display: flex;
    gap: 10px;
    margin-bottom: 1.5rem;
}

.mode-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

@keyframes highlight {
    0% { border-color: var(--gray-300); }
    50% { border-color: var(--primary-color); }
    100% { border-color: var(--gray-300); }
}

.enabled {
    animation: highlight 1s ease;
}

.empty-history {
    padding: 1rem;
    text-align: center;
    color: var(--gray-600);
}
`;
document.head.appendChild(style);

// Add syntax highlighting
if (!window.hljs) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css';
    document.head.appendChild(link);
    
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js';
    document.head.appendChild(script);
}
