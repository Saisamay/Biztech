document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();
  
    const fileInput = document.getElementById('videoFile');
    const resultsSection = document.getElementById('results');
    const videoPreviewSection = document.getElementById('videoPreview');
    const statusElement = document.getElementById('status');
    const detailsElement = document.getElementById('details');
    const videoElement = document.getElementById('uploadedVideo');
  
    const file = fileInput.files[0];
  
    if (file) {
      // Display video preview
      videoElement.src = URL.createObjectURL(file);
      videoPreviewSection.classList.remove('hidden');
  
      // Simulate video analysis (replace with actual API call)
      statusElement.textContent = 'Analyzing...';
      detailsElement.textContent = 'Please wait while we analyze the video.';
  
      setTimeout(() => {
        statusElement.textContent = 'Analysis Complete';
        detailsElement.textContent = 'No suspicious activity detected.';
        resultsSection.classList.remove('hidden');
      }, 3000); // Simulate a 3-second delay for analysis
    } else {
      alert('Please upload a video file.');
    }
  });