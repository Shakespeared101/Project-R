// Event listener for the "Scrape Data" button
document.getElementById("scrapeDataButton").addEventListener("click", function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        const url = tabs[0].url;
        fetch('http://127.0.0.1:5000/scrape', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to scrape data');
            }
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'final.txt';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            alert("Upload the downloaded file to the extension to higlight the dark patterns")
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error: ' + error.message); // Display the error in an alert
        });
    });
});

document.getElementById("summarizeButton").addEventListener("click", function () {
    // Redirect to the summarizing HTML site
    chrome.runtime.sendMessage({ action: "summarize" });
});
document.getElementById("policyCheckerButton").addEventListener("click", function () {
    // Inform the background script to open a new tab for the policy checker
    chrome.runtime.sendMessage({ action: "policyChecker" });
});
document.getElementById("predictButton").addEventListener("click", function () {
    // Inform the background script to open a new tab for the policy checker
    chrome.runtime.sendMessage({ action: "predict" });
});
document.getElementById("FeedbackButton").addEventListener("click", function () {
    // Inform the background script to open a new tab for the policy checker
    chrome.runtime.sendMessage({ action: "feedback" });
});