// Show reversal letter images
const reversalSection = document.getElementById('reversalSection');
const reversalDiv = document.getElementById('reversalImages');

if (data.reversal_images && data.reversal_images.length > 0) {
    reversalSection.style.display = 'block';
    reversalDiv.innerHTML = '';
    
    data.reversal_images.forEach((imgFile, index) => {
        const container = document.createElement('div');
        container.style.cssText = 'text-align:center; background:white; padding:6px; border-radius:6px; border:2px solid #e74c3c;';
        
        const img = document.createElement('img');
        img.src = '/static/uploads/' + imgFile + '?v=' + Date.now();
        img.style.cssText = 'width:80px; height:80px; object-fit:contain; display:block;';
        img.alt = 'Reversal letter ' + (index + 1);
        
        const label = document.createElement('span');
        label.textContent = '#' + (index + 1);
        label.style.cssText = 'font-size:11px; color:#c0392b; font-weight:bold; display:block; margin-top:3px;';
        
        container.appendChild(img);
        container.appendChild(label);
        reversalDiv.appendChild(container);
    });
} else {
    reversalSection.style.display = 'none';

    // NEW: Display per‑letter details in the table (text only)
function displayLetterDetails(data) {
    const container = document.getElementById('letterDetails');
    const tbody = document.getElementById('letterDetailsBody');
    if (data.letter_details && data.letter_details.length > 0) {
        tbody.innerHTML = '';
        data.letter_details.forEach(detail => {
            const row = document.createElement('tr');
            row.style.borderBottom = '1px solid #eaeaea';
            let colour = '#27ae60';   // normal = green
            let label = 'Normal';
            if (detail.type === 'reversal') {
                colour = '#e74c3c';
                label = '🔄 Reversal';
            } else if (detail.type === 'corrected') {
                colour = '#f39c12';
                label = '✏️ Corrected';
            } else {
                label = '✅ Normal';
            }
            row.innerHTML = `
                <td style="padding: 0.5rem 1rem; text-align: center;">${detail.position}</td>
                <td style="padding: 0.5rem 1rem; text-align: center; font-weight: bold;">${detail.expected}</td>
                <td style="padding: 0.5rem 1rem; text-align: center; color: ${colour}; font-weight: bold;">${label}</td>
                <td style="padding: 0.5rem 1rem; text-align: center;">${detail.confidence}%</td>
            `;
            tbody.appendChild(row);
        });
        container.style.display = 'block';
    } else {
        container.style.display = 'none';
    }
}
}