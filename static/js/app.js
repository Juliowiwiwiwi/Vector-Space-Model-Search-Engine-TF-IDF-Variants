/**
 * Vector Space Model Search Engine - Client Controller
 * Author: Devanarayan C S | Roll No: 23FE10CCE00110
 * B.Tech Computer and Communication Engineering (4th Year), Manipal University
 */

document.addEventListener('DOMContentLoaded', () => {
    // Quick sample query filler
    const sampleChips = document.querySelectorAll('.sample-chip');
    const searchInput = document.querySelector('.search-input');
    const searchForm = document.querySelector('.search-form');

    if (sampleChips && searchInput) {
        sampleChips.forEach(chip => {
            chip.addEventListener('click', (e) => {
                e.preventDefault();
                const queryText = chip.getAttribute('data-query') || chip.innerText.trim();
                searchInput.value = queryText;
                searchInput.focus();
                if (searchForm) {
                    searchForm.submit();
                }
            });
        });
    }

    // Keyboard shortcut: '/' to focus search input, 'Escape' to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === '/' && document.activeElement !== searchInput) {
            e.preventDefault();
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        } else if (e.key === 'Escape') {
            closeDocModal();
        }
    });

    // Modal elements
    const modalBackdrop = document.getElementById('docModal');
    const modalTitle = document.getElementById('modalDocTitle');
    const modalAuthors = document.getElementById('modalDocAuthors');
    const modalSource = document.getElementById('modalDocSource');
    const modalTopic = document.getElementById('modalDocTopic');
    const modalWords = document.getElementById('modalDocWords');
    const modalBody = document.getElementById('modalDocContent');
    const modalCloseBtn = document.getElementById('modalCloseBtn');

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', closeDocModal);
    }
    if (modalBackdrop) {
        modalBackdrop.addEventListener('click', (e) => {
            if (e.target === modalBackdrop) {
                closeDocModal();
            }
        });
    }

    function openDocModal(docId) {
        if (!modalBackdrop) return;
        modalBackdrop.classList.add('open');
        modalBody.innerHTML = '<div style="text-align:center; padding: 40px;"><p style="color:var(--text-muted)">Loading document from corpus...</p></div>';

        fetch(`/api/document/${docId}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    modalBody.innerHTML = `<p style="color:var(--red-soft)">Error: ${data.error}</p>`;
                    return;
                }
                modalTitle.innerText = `[Doc #${data.doc_id}] ${data.title}`;
                modalAuthors.innerText = data.authors;
                modalSource.innerText = `${data.source} (${data.year})`;
                modalTopic.innerText = data.topic;
                modalWords.innerText = `${data.word_count.toLocaleString()} words`;

                modalBody.innerHTML = `
                    <div style="margin-bottom: 20px;">
                        <h4 style="color:var(--green-soft); font-size:0.9rem; text-transform:uppercase; margin-bottom:6px;">Abstract / Executive Summary</h4>
                        <div style="background:rgba(0,0,0,0.3); padding:14px; border-radius:8px; border-left:3px solid var(--green-soft); color:var(--text-main); font-size:0.95rem;">
                            ${escapeHtml(data.abstract)}
                        </div>
                    </div>
                    <div>
                        <h4 style="color:var(--red-soft); font-size:0.9rem; text-transform:uppercase; margin-bottom:8px;">Document Excerpt & Header</h4>
                        <pre style="white-space: pre-wrap; font-family:inherit; background:var(--bg-deep); padding:16px; border-radius:8px; border:1px solid var(--border-subtle); font-size:0.85rem; max-height:350px; overflow-y:auto; color:var(--text-muted);">${escapeHtml(data.content_excerpt)}</pre>
                    </div>
                `;
            })
            .catch(err => {
                modalBody.innerHTML = `<p style="color:var(--red-soft)">Failed to load document: ${err}</p>`;
            });
    }

    function closeDocModal() {
        if (modalBackdrop) {
            modalBackdrop.classList.remove('open');
        }
    }

    // Attach view doc buttons
    document.querySelectorAll('.btn-view-doc').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const docId = btn.getAttribute('data-doc-id');
            if (docId) {
                openDocModal(docId);
            }
        });
    });

    function escapeHtml(text) {
        if (!text) return '';
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
});

