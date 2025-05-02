/**
 * Auto-Learning Website Chatbot
 * Embeddable script that can be added to any website.
 */

(function() {
    // Configuration
    const config = {
        apiBaseUrl: window.location.origin,
        chatbotPosition: 'bottom-right',
        primaryColor: '#4a6fa5',
        chatTitle: 'Website Assistant',
        placeholderText: 'Ask me anything about this website...',
        welcomeMessage: 'Hello! I\'m your website assistant. Ask me anything about this site.'
    };

    // Create and inject CSS
    function injectStyles() {
        const styleEl = document.createElement('style');
        styleEl.innerHTML = `
            .chatbot-container {
                position: fixed;
                ${config.chatbotPosition.includes('bottom') ? 'bottom: 20px;' : 'top: 20px;'}
                ${config.chatbotPosition.includes('right') ? 'right: 20px;' : 'left: 20px;'}
                z-index: 9999;
                display: flex;
                flex-direction: column;
                width: 350px;
                max-width: 90vw;
                height: 480px;
                max-height: 80vh;
                background: white;
                border-radius: 10px;
                box-shadow: 0 5px 25px rgba(0, 0, 0, 0.2);
                overflow: hidden;
                transition: transform 0.3s, opacity 0.3s;
                transform: translateY(100%);
                opacity: 0;
                font-family: Arial, sans-serif;
            }
            
            .chatbot-container.open {
                transform: translateY(0);
                opacity: 1;
            }
            
            .chatbot-header {
                background-color: ${config.primaryColor};
                color: white;
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .chatbot-header h3 {
                margin: 0;
                font-size: 16px;
            }
            
            .chatbot-close {
                background: none;
                border: none;
                color: white;
                font-size: 20px;
                cursor: pointer;
                padding: 0;
            }
            
            .chatbot-messages {
                flex: 1;
                padding: 15px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
            }
            
            .chatbot-message {
                max-width: 80%;
                padding: 10px 15px;
                border-radius: 18px;
                margin-bottom: 10px;
                line-height: 1.4;
                font-size: 14px;
                word-wrap: break-word;
            }
            
            .chatbot-message.bot {
                background-color: #f1f1f1;
                align-self: flex-start;
                border-bottom-left-radius: 4px;
            }
            
            .chatbot-message.user {
                background-color: ${config.primaryColor};
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 4px;
            }
            
            .chatbot-input-container {
                display: flex;
                padding: 10px;
                border-top: 1px solid #eee;
            }
            
            .chatbot-input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 20px;
                font-size: 14px;
            }
            
            .chatbot-send {
                background-color: ${config.primaryColor};
                color: white;
                border: none;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                margin-left: 10px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .chatbot-send svg {
                width: 18px;
                height: 18px;
            }
            
            .chatbot-button {
                position: fixed;
                ${config.chatbotPosition.includes('bottom') ? 'bottom: 20px;' : 'top: 20px;'}
                ${config.chatbotPosition.includes('right') ? 'right: 20px;' : 'left: 20px;'}
                background-color: ${config.primaryColor};
                color: white;
                border: none;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                z-index: 9998;
            }
            
            .chatbot-typing {
                display: flex;
                padding: 10px;
                align-self: flex-start;
            }
            
            .chatbot-typing-bubble {
                background-color: #e6e6e6;
                border-radius: 50%;
                display: inline-block;
                height: 8px;
                width: 8px;
                margin: 0 1px;
                animation: typing 1s infinite ease-in-out;
            }
            
            .chatbot-typing-bubble:nth-child(1) {
                animation-delay: 0s;
            }
            
            .chatbot-typing-bubble:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .chatbot-typing-bubble:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
                100% { transform: translateY(0); }
            }
            
            .chatbot-sources {
                font-size: 11px;
                margin-top: 5px;
                color: #666;
            }
            
            .chatbot-sources a {
                color: ${config.primaryColor};
                text-decoration: none;
            }
            
            .chatbot-sources a:hover {
                text-decoration: underline;
            }
            
            @media (max-width: 480px) {
                .chatbot-container {
                    width: 100%;
                    max-width: 100%;
                    height: 100%;
                    max-height: 100%;
                    top: 0;
                    right: 0;
                    bottom: 0;
                    left: 0;
                    border-radius: 0;
                }
            }
        `;
        document.head.appendChild(styleEl);
    }

    // Create chatbot UI
    function createChatbotUI() {
        // Create chat button
        const chatButton = document.createElement('button');
        chatButton.className = 'chatbot-button';
        chatButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
        `;
        document.body.appendChild(chatButton);

        // Create chat container
        const chatContainer = document.createElement('div');
        chatContainer.className = 'chatbot-container';
        chatContainer.innerHTML = `
            <div class="chatbot-header">
                <h3>${config.chatTitle}</h3>
                <button class="chatbot-close">&times;</button>
            </div>
            <div class="chatbot-messages"></div>
            <div class="chatbot-input-container">
                <input type="text" class="chatbot-input" placeholder="${config.placeholderText}">
                <button class="chatbot-send">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
        `;
        document.body.appendChild(chatContainer);

        // Get elements
        const messagesContainer = chatContainer.querySelector('.chatbot-messages');
        const inputField = chatContainer.querySelector('.chatbot-input');
        const sendButton = chatContainer.querySelector('.chatbot-send');
        const closeButton = chatContainer.querySelector('.chatbot-close');

        // Add welcome message
        addBotMessage(config.welcomeMessage);

        // Add event listeners
        chatButton.addEventListener('click', () => {
            chatContainer.classList.add('open');
            chatButton.style.display = 'none';
            // Auto-scrape the current website if needed
            checkAndScrapeWebsite();
        });

        closeButton.addEventListener('click', () => {
            chatContainer.classList.remove('open');
            chatButton.style.display = 'flex';
        });

        sendButton.addEventListener('click', sendMessage);
        inputField.addEventListener('keypress', e => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Helper functions
        function sendMessage() {
            const message = inputField.value.trim();
            if (!message) return;

            // Add user message
            addUserMessage(message);
            inputField.value = '';

            // Show typing indicator
            const typingIndicator = addTypingIndicator();

            // Send to API
            fetch(`${config.apiBaseUrl}/api/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: message })
            })
            .then(response => response.json())
            .then(data => {
                // Remove typing indicator
                typingIndicator.remove();
                // Add bot response
                addBotMessage(data.answer, data.contexts);
            })
            .catch(error => {
                console.error('Error querying API:', error);
                typingIndicator.remove();
                addBotMessage('Sorry, I encountered an error. Please try again.');
            });
        }

        function addUserMessage(message) {
            const messageEl = document.createElement('div');
            messageEl.className = 'chatbot-message user';
            messageEl.textContent = message;
            messagesContainer.appendChild(messageEl);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addBotMessage(message, contexts = []) {
            const messageEl = document.createElement('div');
            messageEl.className = 'chatbot-message bot';
            
            // Convert markdown-like formatting
            let formattedMessage = message
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
            
            messageEl.innerHTML = formattedMessage;
            
            // Add sources if available
            if (contexts && contexts.length > 0) {
                const sourcesEl = document.createElement('div');
                sourcesEl.className = 'chatbot-sources';
                
                // Get unique sources
                const uniqueSources = {};
                contexts.forEach(context => {
                    if (context.source && context.title) {
                        uniqueSources[context.source] = context.title;
                    }
                });
                
                const sourcesList = Object.entries(uniqueSources)
                    .map(([url, title]) => `<a href="${url}" target="_blank">${title}</a>`)
                    .join(', ');
                
                sourcesEl.innerHTML = `Sources: ${sourcesList}`;
                messageEl.appendChild(sourcesEl);
            }
            
            messagesContainer.appendChild(messageEl);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addTypingIndicator() {
            const typingEl = document.createElement('div');
            typingEl.className = 'chatbot-typing';
            typingEl.innerHTML = `
                <div class="chatbot-typing-bubble"></div>
                <div class="chatbot-typing-bubble"></div>
                <div class="chatbot-typing-bubble"></div>
            `;
            messagesContainer.appendChild(typingEl);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            return typingEl;
        }
    }

    // Check if the current website needs scraping
    function checkAndScrapeWebsite() {
        // Get current website URL
        const currentUrl = window.location.origin;
        
        // Check if this website is already in the database
        fetch(`${config.apiBaseUrl}/api/status`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ok' && data.database.document_count === 0) {
                    // No documents in database, scrape this website
                    scrapeWebsite(currentUrl);
                }
            })
            .catch(error => {
                console.error('Error checking database status:', error);
            });
    }

    // Scrape the current website
    function scrapeWebsite(url) {
        fetch(`${config.apiBaseUrl}/api/scrape`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Started scraping website:', data);
        })
        .catch(error => {
            console.error('Error scraping website:', error);
        });
    }

    // Initialize
    function init() {
        injectStyles();
        createChatbotUI();
    }

    // Run initialization when the DOM is fully loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
