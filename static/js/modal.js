/**
 * Simplified chat history modal functionality
 * Completely rewritten to use direct DOM manipulation with inline styles
 */

// Define a variable to store the current chat history data
let chatHistoryData = [];

// Function to hide the history overlay
function hideHistoryOverlay() {
    const overlay = document.getElementById('history-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Function to show the history overlay
function showHistoryOverlay() {
    const overlay = document.getElementById('history-overlay');
    if (overlay) {
        overlay.style.display = 'block';
    }
}

// Function to populate the history list
function populateHistoryList(history) {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;
    
    historyList.innerHTML = ''; // Clear previous content
    
    if (history && history.length > 0) {
        history.forEach(item => {
            const div = document.createElement('div');
            div.style.padding = '10px';
            div.style.margin = '5px 0';
            div.style.borderRadius = '4px';
            div.style.backgroundColor = item.role === 'user' ? '#f0f7ff' : '#f5f5f5';
            
            const roleLabel = item.role === 'user' ? 'You' : 'Bot';
            const message = typeof item.message === 'string' 
                ? item.message 
                : JSON.stringify(item.message);
            
            div.innerHTML = `<strong>${roleLabel}:</strong> ${message}`;
            historyList.appendChild(div);
        });
    } else {
        const emptyDiv = document.createElement('div');
        emptyDiv.style.padding = '20px';
        emptyDiv.style.textAlign = 'center';
        emptyDiv.style.color = '#666';
        emptyDiv.textContent = 'No chat history available.';
        historyList.appendChild(emptyDiv);
    }
}

// Initialize everything when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Make sure the overlay is hidden on page load
    hideHistoryOverlay();
    
    // Add click handler to the history button
    const historyBtn = document.getElementById('history-btn');
    if (historyBtn) {
        historyBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Fetch the chat history
            fetch('/api/chat_history')
                .then(response => response.json())
                .then(data => {
                    chatHistoryData = data.history || [];
                    populateHistoryList(chatHistoryData);
                    showHistoryOverlay();
                })
                .catch(error => {
                    console.error('Error fetching chat history:', error);
                    chatHistoryData = [];
                    populateHistoryList([]);
                    showHistoryOverlay();
                });
        });
    }
    
    // Add click handler to the close button
    const closeBtn = document.getElementById('close-history-btn');
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            hideHistoryOverlay();
        });
    }
    
    // Close when clicking on the overlay background
    const overlay = document.getElementById('history-overlay');
    if (overlay) {
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                hideHistoryOverlay();
            }
        });
    }
    
    // Close when pressing Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            hideHistoryOverlay();
        }
    });
});

// Ensure the overlay is hidden when the page loads
window.addEventListener('load', hideHistoryOverlay);

// For cases where the DOM is already loaded
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    hideHistoryOverlay();
}
