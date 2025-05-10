/**
 * Modal functionality for the chat history
 * This is a standalone script that handles the modal without relying on the main chat.js file
 */

// Define global function for hiding modal that will be available immediately
function hideHistoryModal() {
    const modal = document.getElementById('history-modal');
    if (modal) {
        console.log('Hiding modal via global function');
        modal.classList.add('hidden');
    }
}

// Function to initialize close button - can be called multiple times
function initCloseButton() {
    const closeButton = document.getElementById('close-history');
    if (closeButton) {
        // Remove any existing event listeners by cloning and replacing
        const newCloseBtn = closeButton.cloneNode(true);
        closeButton.parentNode.replaceChild(newCloseBtn, closeButton);
        
        // Add new event listener
        newCloseBtn.addEventListener('click', function(event) {
            console.log('Close button clicked');
            event.preventDefault();
            event.stopPropagation();
            hideHistoryModal();
            return false;
        });
        
        // Add additional cursor style to ensure it looks clickable
        newCloseBtn.style.cursor = 'pointer';
        console.log('Close button initialized with event listener');
    } else {
        console.error('Close button (#close-history) not found when initializing');
    }
}

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get the modal and history button
    const modal = document.getElementById('history-modal');
    const historyButton = document.getElementById('history-btn');
    
    // Initialize close button
    initCloseButton();
    
    // Add click event to open the modal
    if (historyButton) {
        historyButton.addEventListener('click', function() {
            console.log('History button clicked');
            fetch('/api/chat_history')
                .then(response => response.json())
                .then(data => {
                    const historyList = document.getElementById('history-list');
                    if (!historyList) {
                        console.error('History list element not found');
                        return;
                    }
                    historyList.innerHTML = ''; // Clear previous history
                    
                    if (data.history && data.history.length > 0) {
                        data.history.forEach(item => {
                            const div = document.createElement('div');
                            div.className = 'history-item ' + item.role;
                            
                            const roleIcon = item.role === 'user' 
                                ? '<i class="fas fa-user"></i>' 
                                : '<i class="fas fa-robot"></i>';
                            
                            const roleLabel = item.role === 'user' ? 'You' : 'Bot';
                            const message = typeof item.message === 'string' 
                                ? item.message 
                                : JSON.stringify(item.message);
                            
                            div.innerHTML = `<strong>${roleIcon} ${roleLabel}:</strong> ${message}`;
                            historyList.appendChild(div);
                        });
                    } else {
                        historyList.innerHTML = '<div class="empty-history"><i class="fas fa-info-circle"></i> No chat history.</div>';
                    }
                    
                    // Show modal
                    if (modal) {
                        console.log('Showing modal');
                        modal.classList.remove('hidden');
                        // Re-initialize close button after modal is shown
                        setTimeout(initCloseButton, 100);
                    } else {
                        console.error('Modal element not found when trying to show');
                    }
                })
                .catch(error => {
                    console.error('Error loading chat history:', error);
                    const historyList = document.getElementById('history-list');
                    if (historyList) {
                        historyList.innerHTML = '<div class="empty-history error"><i class="fas fa-exclamation-triangle"></i> Error loading history.</div>';
                    }
                });
        });
    } else {
        console.log('History button not found');
    }

    // Close modal when clicking outside of it
    window.addEventListener('click', function(event) {
        if (modal && event.target === modal) {
            console.log('Clicked outside modal content');
            hideHistoryModal();
        }
    });

    // Close modal when pressing Escape key
    document.addEventListener('keydown', function(event) {
        if (modal && !modal.classList.contains('hidden') && event.key === 'Escape') {
            console.log('Escape key pressed');
            hideHistoryModal();
        }
    });
});

// Backup initialization for cases where DOM might be already loaded
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(initCloseButton, 100);
}
