document.addEventListener('DOMContentLoaded', function() {
    let currentLevel = 'easy';
    
    document.querySelectorAll('.level-pill').forEach(pill => {
        pill.addEventListener('click', function() {
            document.querySelectorAll('.level-pill').forEach(p => p.classList.remove('active'));
            this.classList.add('active');
            currentLevel = this.dataset.level;
            loadNewPhrase();
        });
    });
    
    document.getElementById('newPhraseBtn')?.addEventListener('click', loadNewPhrase);
    
    async function loadNewPhrase() {
        const response = await fetch(`/api/get_phrase?level=${currentLevel}`);
        const data = await response.json();
        document.getElementById('phraseDisplay').textContent = data.phrase;
        document.getElementById('expectedPhrase').value = data.phrase;
    }
    
    loadNewPhrase();
});