chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === "summarize") {
    // Implement your text summarization logic here
    let summarizedText = "This is a sample summary.";
    sendResponse({ summary: summarizedText });
  }
  // Add other actions as needed
});
function highlightDarkPatterns(results) {
  // Read the contents of the output_text.txt file
  fetch(chrome.extension.getURL('output_text.txt'))
      .then(response => response.text())
      .then(textContent => {
          // Split the text content into lines
          const lines = textContent.split('\n');

          // Iterate through the lines and highlight based on the results
          results.forEach(entry => {
              const linesToHighlight = findLinesContainingText(lines, entry.text);
              linesToHighlight.forEach(lineIndex => {
                  // Apply a background color to highlight the line
                  highlightLine(lineIndex);
              });
          });
      })
      .catch(error => {
          console.error('Error reading output_text.txt:', error);
      });
}

function findLinesContainingText(lines, text) {
  // Use a simple text-based search to find line indices containing the specified text
  const matchingLines = lines.reduce((acc, line, index) => {
      if (line.includes(text)) {
          acc.push(index);
      }
      return acc;
  }, []);
  return matchingLines;
}

function highlightLine(lineIndex) {
  // Apply a background color to highlight the line
  const lineElement = document.querySelector(`#line-${lineIndex}`);
  if (lineElement) {
      lineElement.style.backgroundColor = 'rgba(255, 0, 0, 0.3)'; // Red with 30% opacity
  }
}

// You can use a content script to add line numbers to each line in the DOM
// This function is just for demonstration purposes
function addLineNumbersToDOM() {
  const lines = document.querySelectorAll('pre, code, p'); // Adjust selector based on your webpage's structure
  lines.forEach((line, index) => {
      line.id = `line-${index}`;
  });
}

