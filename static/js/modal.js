/**
 * Modal functionality for the chat history
 * This is a standalone script that handles the modal without relying on the main chat.js file
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get the modal and close button
    const modal = document.getElementById('history-modal');
    const closeButton = document.getElementById('close-history');
    const historyButton = document.getElementById('history-btn');
    
    // Function to hide modal
    function hideModal() {
        if (modal) {
            modal.classList.add('hidden');
        }
    }
    
    // Add click event to close button
    if (closeButton) {
        // First remove any existing event listeners
        closeButton.replaceWith(closeButton.cloneNode(true));
        
        // Get the fresh button without event listeners
        const newCloseButton = document.getElementById('close-history');
        
        // Add click event
        newCloseButton.addEventListener('click', function(event) {
            console.log('Close button clicked');
            event.preventDefault();
            event.stopPropagation();
            hideModal();
        });
    }
    
    // Add click event to open the modal
    if (historyButton) {
        historyButton.addEventListener('click', function() {
            fetch('/api/chat_history')
                .then(response => response.json())
                .then(data => {
                    const historyList = document.getElementById('history-list');
                    historyList.innerHTML = '';
                    
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
                        modal.classList.remove('hidden');
                    }
                })
                .catch(error => {
                    console.error('Error loading chat history:', error);
                });
        });
    }
    
    // Close modal when clicking outside of it
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            hideModal();
        }
    });
    
    // Close modal when pressing Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            hideModal();
        }
    });
});

// Add direct onclick attribute to close button as a backup
window.onload = function() {
    const closeButton = document.getElementById('close-history');
    if (closeButton) {
        closeButton.setAttribute('onclick', 'document.getElementById("history-modal").classList.add("hidden");');
    }
};
