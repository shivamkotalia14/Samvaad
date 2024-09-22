		document.addEventListener('DOMContentLoaded', function() {
			const chatbotToggle = document.getElementById('chatbot-toggle');
			const chatbotContent = document.getElementById('chatbot-content');
			const chatbotClose = document.getElementById('chatbot-close');
			const chatbotTextInput = document.getElementById('chatbot-input');
			const chatWindow = document.querySelector('.chatbot-messages');
			let text = '';
			let video = null;
			let videoName = '';
			let mediaRecorder;
			let chunks = [];
		
			// Automatically open chatbot on page load
			chatbotContent.style.display = 'block'; // Ensure the chatbot is open
		
			// Toggle chatbot visibility
			chatbotToggle.addEventListener('click', function() {
				chatbotContent.style.display = chatbotContent.style.display === 'block' ? 'none' : 'block';
			});
		
			chatbotClose.addEventListener('click', function() {
				chatbotContent.style.display = 'none';
			});
		
			// Handle text change
			function handleTextChange(e) {
				text = e.target.value;
			}
		
			// Function to add chat message to the chat window
			function addChatMessage(message, type) {
				const messageDiv = document.createElement('div');
				messageDiv.classList.add('chat-message', type === 'user' ? 'user-message' : 'bot-message');
				messageDiv.textContent = message;
				chatWindow.appendChild(messageDiv);
				chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to bottom
			}
		
			// Handle form submission
			async function handleSubmit() {
				const formData = new FormData();
				if (text) formData.append('text', text);
		
				// Add user message to chat window
				addChatMessage(text, 'user');
		
				try {
					const response = await fetch('http://localhost:5000/upload', {
						method: 'POST',
						body: formData,
					});
					const data = await response.json();
		
					// Add bot response to chat window
					addChatMessage(data.message, 'bot');
		
					// Clear the inputs
					text = '';
					chatbotTextInput.value = '';
				} catch (error) {
					console.error('Error uploading data:', error);
					addChatMessage('Error processing your request. Please try again.', 'bot');
				}
			}
		
			// Attach event listeners
			chatbotTextInput.addEventListener('input', handleTextChange);
			document.getElementById('chatbot-submit').addEventListener('click', handleSubmit);
		});
