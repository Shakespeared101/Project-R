chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.action === "summarize") {
      fetch('http://localhost:5000/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: request.text }),  // Pass the selected text to the server
      })
      .then(response => response.json())
      .then(data => {
        console.log("Summary:", data.summary);
        sendResponse({ summary: data.summary });
      })
      .catch(error => {
        console.error('Error:', error);
        sendResponse({ summary: 'Error occurred during summarization.' });
      });
      return true; // This ensures that the sendResponse is asynchronous
    }
    // Add other actions as needed
  });
  chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.action === "summarize") {
        // Open a new tab only when summarization is requested
        chrome.tabs.create({ url: 'http://localhost:5000' });
    }
    else if (request.action === "policyChecker") {
      // Open a new tab for the policy checker
      chrome.tabs.create({ url: 'http://localhost:5000/compare' });
  }
  else if (request.action === "predict") {
    const reviewText = encodeURIComponent(request.review); // Encode review text
    const rating = encodeURIComponent(request.rating); // Encode rating
    const predictionUrl = `http://localhost:5000/predict`;

    // Open a new tab for the prediction URL
    chrome.tabs.create({ url: predictionUrl });
}
else if (request.action === "feedback") {
  const predictionUrl = 'http://127.0.0.1:5000/feedback';

  // Open a new tab for the prediction URL
  chrome.tabs.create({ url: predictionUrl });
}
    // Add other actions as needed
}); 



  