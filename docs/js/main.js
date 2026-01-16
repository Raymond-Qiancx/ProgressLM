// ProgressLM - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // BibTeX Copy Functionality
    initCopyButton();

    // Smooth scroll for anchor links
    initSmoothScroll();

    // Optional: Fetch GitHub stats
    // fetchGitHubStats();
});

/**
 * Initialize BibTeX copy button functionality
 */
function initCopyButton() {
    const copyBtn = document.getElementById('copy-btn');
    const bibtexContent = document.getElementById('bibtex-content');

    if (copyBtn && bibtexContent) {
        copyBtn.addEventListener('click', async function() {
            try {
                await navigator.clipboard.writeText(bibtexContent.textContent);

                // Visual feedback
                const originalHTML = copyBtn.innerHTML;
                copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                copyBtn.classList.add('copied');

                // Reset after 2 seconds
                setTimeout(function() {
                    copyBtn.innerHTML = originalHTML;
                    copyBtn.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
                // Fallback for older browsers
                fallbackCopy(bibtexContent.textContent);
            }
        });
    }
}

/**
 * Fallback copy function for browsers without clipboard API
 */
function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
        document.execCommand('copy');
        const copyBtn = document.getElementById('copy-btn');
        if (copyBtn) {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            copyBtn.classList.add('copied');
            setTimeout(function() {
                copyBtn.innerHTML = originalHTML;
                copyBtn.classList.remove('copied');
            }, 2000);
        }
    } catch (err) {
        console.error('Fallback copy failed: ', err);
    }

    document.body.removeChild(textArea);
}

/**
 * Initialize smooth scrolling for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Optional: Fetch GitHub repository stats
 * Uncomment and modify the repo path to enable
 */
function fetchGitHubStats() {
    const repo = 'Raymond-Qiancx/ProgressLM';
    const apiUrl = `https://api.github.com/repos/${repo}`;

    fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            const starsEl = document.getElementById('github-stars');
            const forksEl = document.getElementById('github-forks');

            if (starsEl) {
                starsEl.textContent = formatNumber(data.stargazers_count);
            }
            if (forksEl) {
                forksEl.textContent = formatNumber(data.forks_count);
            }
        })
        .catch(err => {
            console.log('Could not fetch GitHub stats:', err);
        });
}

/**
 * Format large numbers with K suffix
 */
function formatNumber(num) {
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}
