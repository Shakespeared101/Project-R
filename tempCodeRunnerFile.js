document.getElementById("summarizeButton").addEventListener("click", function () {
    // Redirect to the summarizing HTML site
    chrome.runtime.sendMessage({ action: "summarize" });
});
