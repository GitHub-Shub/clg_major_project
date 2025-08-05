async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const query = userInput.value.trim();
    if (!query) return;

    const chatMessages = document.getElementById('chatMessages');

    // Display user message
    appendMessage(query, 'user-message');
    userInput.value = ''; // Clear input field

    // Display loading indicator
    const loadingIndicator = appendMessage('...', 'bot-message loading');
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query }),
        });

        chatMessages.removeChild(loadingIndicator); // Remove loading indicator

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.response || `HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        displayBotResponse(data);

    } catch (error) {
        chatMessages.removeChild(loadingIndicator);
        appendMessage(`Error: ${error.message}`, 'bot-message error');
    }
}

function appendMessage(text, className) {
    const chatMessages = document.getElementById('chatMessages');
    const messageElement = document.createElement('div');
    messageElement.className = `message ${className}`;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<p>${text}</p>`; // Use innerHTML to allow for basic formatting if needed
    
    messageElement.appendChild(content);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageElement;
}

function displayBotResponse(data) {
    const chatMessages = document.getElementById('chatMessages');
    const messageElement = document.createElement('div');
    messageElement.className = 'message bot-message';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<p>${data.response.replace(/\n/g, '<br>')}</p>`; // Main response

    // Create and append the explanation section
    if (data.explanation) {
        const explanationHTML = createExplanationHTML(data.explanation);
        content.appendChild(explanationHTML);
    }

    messageElement.appendChild(content);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createExplanationHTML(explanation) {
    const detailsContainer = document.createElement('details');
    detailsContainer.className = 'explanation-details';

    const summary = document.createElement('summary');
    summary.innerHTML = '<span>Show Explanation</span><i class="fa-solid fa-chevron-down"></i>';
    detailsContainer.appendChild(summary);

    const wrapper = document.createElement('div');
    wrapper.className = 'explanation-content';

    const confidence = explanation.confidence_explanation;
    const cacheInfo = explanation.cache_explanation;
    const legal = explanation.legal_analysis;
    const sources = explanation.source_attribution;

    // Confidence Section
    const confidenceColor = confidence.confidence_category === 'high' ? 'green' : (confidence.confidence_category === 'medium' ? 'orange' : 'red');
    wrapper.innerHTML += `
        <h4><i class="fa-solid fa-check-double"></i> Confidence Analysis</h4>
        <div class="confidence-bar">
            <div class="confidence-level ${confidenceColor}" style="width: ${confidence.overall_confidence * 100}%;">
                ${(confidence.overall_confidence * 100).toFixed(1)}%
            </div>
        </div>
        <p class="metadata">
            <strong>Method:</strong> ${cacheInfo.why_this_cache_level}
        </p>
    `;

    // Source Attribution Section
    if (sources && sources.length > 0) {
        wrapper.innerHTML += `<h4><i class="fa-solid fa-book-open"></i> Source Attribution</h4>`;
        sources.slice(0, 3).forEach((source, index) => {
            wrapper.innerHTML += `
                <div class="source-card">
                    <p><strong>Source ${index + 1}:</strong> ${source.relevance_explanation}</p>
                    <p class="metadata">
                        <strong>Similarity:</strong> ${source.similarity_score.toFixed(3)} | 
                        <strong>Legal Sections:</strong> ${source.legal_sections.join(', ') || 'N/A'}
                    </p>
                    <p class="source-text">"${source.text}"</p>
                </div>
            `;
        });
    }

    // Legal Analysis Section
    if (legal && legal.legal_sections_referenced.length > 0) {
        wrapper.innerHTML += `
            <h4><i class="fa-solid fa-gavel"></i> Legal Analysis</h4>
            <p class="metadata">
                <strong>Sections Referenced:</strong> ${legal.legal_sections_referenced.join(', ')}<br>
                <strong>Penalties Mentioned:</strong> ${legal.penalty_amounts_mentioned.join(', ') || 'None'}<br>
                <strong>Legal Certainty:</strong> <span class="certainty-${legal.legal_certainty}">${legal.legal_certainty}</span>
            </p>
        `;
    }
    
    // Knowledge Gaps
    const gaps = explanation.knowledge_gaps;
    if (gaps && gaps.recommendations_for_clarification.length > 0) {
         wrapper.innerHTML += `
            <h4><i class="fa-solid fa-lightbulb"></i> Recommendations</h4>
            <ul class="recommendations-list">
                ${gaps.recommendations_for_clarification.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;
    }


    detailsContainer.appendChild(wrapper);
    return detailsContainer;
}


document.getElementById('userInput').addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
