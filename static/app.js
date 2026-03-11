// ===== File input display =====
document.querySelectorAll('input[type="file"]').forEach((input) => {
    const box = input.closest('.upload-box');
    const nameDisplay = box.querySelector('.file-name');

    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            nameDisplay.textContent = input.files[0].name;
            box.classList.add('has-file');
        } else {
            nameDisplay.textContent = 'No file chosen';
            box.classList.remove('has-file');
        }
    });
});

// ===== Form submission =====
const form = document.getElementById('predict-form');
const submitBtn = document.getElementById('submit-btn');
const btnText = submitBtn.querySelector('.btn-text');
const btnLoading = submitBtn.querySelector('.btn-loading');
const resultSection = document.getElementById('result');
const resultMessage = document.getElementById('result-message');
const downloadLink = document.getElementById('download-link');
const errorSection = document.getElementById('error');
const errorMessage = document.getElementById('error-message');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Reset state
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';

    // Show loading
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';

    try {
        const formData = new FormData(form);

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            resultMessage.textContent = data.message;
            downloadLink.href = data.download_url;
            downloadLink.download = data.filename;
            resultSection.style.display = 'block';
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            const detail = data.detail || data.message || 'An unexpected error occurred.';
            errorMessage.textContent = detail;
            errorSection.style.display = 'block';
            errorSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    } catch (err) {
        errorMessage.textContent = 'Network error. Please check your connection and try again.';
        errorSection.style.display = 'block';
        errorSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } finally {
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
});
