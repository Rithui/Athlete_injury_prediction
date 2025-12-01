/**
 * RecoverAI - Main JavaScript
 * Athlete Injury Recovery Prediction System
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeForm();
    initializeBMICalculator();
    initializeSmoothScroll();
});

/**
 * Initialize the prediction form
 */
function initializeForm() {
    const form = document.getElementById('injuryForm');
    if (!form) return;

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const submitButton = form.querySelector('.submit-button');
        submitButton.classList.add('loading');
        submitButton.disabled = true;

        try {
            // Collect form data
            const formData = {
                age: document.getElementById('age').value,
                gender: document.getElementById('gender').value,
                height: document.getElementById('height').value,
                weight: document.getElementById('weight').value,
                bmi: document.getElementById('bmi').value,
                training_frequency: document.getElementById('training_frequency').value,
                training_duration: document.getElementById('training_duration').value,
                warmup_time: document.getElementById('warmup_time').value,
                sleep_hours: document.getElementById('sleep_hours').value,
                flexibility_score: document.getElementById('flexibility_score').value,
                muscle_asymmetry: document.getElementById('muscle_asymmetry').value,
                recovery_time: document.getElementById('recovery_time').value,
                injury_history: document.getElementById('injury_history').value,
                stress_level: document.getElementById('stress_level').value,
                training_intensity: document.getElementById('training_intensity').value
            };

            // Validate all fields are filled
            for (const [key, value] of Object.entries(formData)) {
                if (!value || value === '') {
                    throw new Error(`Please fill in all fields. Missing: ${key.replace(/_/g, ' ')}`);
                }
            }

            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                displayResults(result);
            } else {
                throw new Error(result.error || 'Prediction failed');
            }

        } catch (error) {
            showNotification(error.message, 'error');
        } finally {
            submitButton.classList.remove('loading');
            submitButton.disabled = false;
        }
    });

    // Reset button handler
    form.addEventListener('reset', function() {
        const resultsSection = document.getElementById('results');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
    });
}

/**
 * Initialize BMI auto-calculator
 */
function initializeBMICalculator() {
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiInput = document.getElementById('bmi');

    if (!heightInput || !weightInput || !bmiInput) return;

    function calculateBMI() {
        const height = parseFloat(heightInput.value);
        const weight = parseFloat(weightInput.value);

        if (height && weight && height > 0) {
            const heightInMeters = height / 100;
            const bmi = weight / (heightInMeters * heightInMeters);
            bmiInput.value = bmi.toFixed(1);
        }
    }

    heightInput.addEventListener('input', calculateBMI);
    weightInput.addEventListener('input', calculateBMI);
}

/**
 * Display prediction results
 */
function displayResults(result) {
    const resultsSection = document.getElementById('results');
    if (!resultsSection) return;

    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Update outlook card
    const outlookCard = document.getElementById('outlookCard');
    outlookCard.className = `outlook-card ${result.outlook_class}`;
    
    document.getElementById('recoveryOutlook').textContent = result.recovery_outlook;
    document.getElementById('recoveryTimeline').textContent = `Estimated: ${result.estimated_recovery_days}`;
    document.getElementById('confidenceValue').textContent = `${result.model_confidence}%`;

    // Update circular gauges
    const quickGauge = document.getElementById('quickGauge');
    const extendedGauge = document.getElementById('extendedGauge');
    const circumference = 2 * Math.PI * 45; // 283
    
    // Animate quick recovery gauge
    const quickOffset = circumference - (result.quick_recovery_probability / 100 * circumference);
    quickGauge.style.transition = 'stroke-dashoffset 1.5s ease';
    quickGauge.style.strokeDashoffset = quickOffset;
    
    // Animate extended recovery gauge
    const extendedOffset = circumference - (result.extended_recovery_probability / 100 * circumference);
    extendedGauge.style.transition = 'stroke-dashoffset 1.5s ease';
    extendedGauge.style.strokeDashoffset = extendedOffset;

    // Update probability values
    animateValue(document.getElementById('quickRecoveryProb'), 0, result.quick_recovery_probability, 1500, '%');
    animateValue(document.getElementById('extendedRecoveryProb'), 0, result.extended_recovery_probability, 1500, '%');

    // Update analysis grid
    const analysisGrid = document.getElementById('analysisGrid');
    analysisGrid.innerHTML = '';
    
    const analysisLabels = {
        'training_load_status': 'Training Load',
        'sleep_status': 'Sleep Quality',
        'flexibility_status': 'Flexibility',
        'stress_status': 'Stress Level',
        'injury_history_impact': 'Injury History'
    };
    
    for (const [key, value] of Object.entries(result.analysis)) {
        const item = document.createElement('div');
        item.className = 'analysis-item';
        
        const valueClass = value.toLowerCase().replace(' ', '-');
        
        item.innerHTML = `
            <div class="label">${analysisLabels[key] || key}</div>
            <div class="value ${valueClass}">${value}</div>
        `;
        analysisGrid.appendChild(item);
    }

    // Update recommendations
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';
    
    result.recommendations.forEach((rec, index) => {
        const li = document.createElement('li');
        
        // Add warning/danger class for certain recommendations
        if (rec.includes('⚠️') || rec.includes('Strongly') || rec.includes('immediately')) {
            li.className = 'danger';
        } else if (rec.includes('Consider') || rec.includes('Monitor')) {
            li.className = 'warning';
        }
        
        li.textContent = rec;
        li.style.animation = `fadeInUp 0.4s ease ${index * 0.1}s both`;
        recommendationsList.appendChild(li);
    });
}

/**
 * Animate numeric value
 */
function animateValue(element, start, end, duration, suffix = '') {
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = start + (end - start) * easeOutQuart;
        
        element.textContent = current.toFixed(1) + suffix;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Remove existing notification
    const existing = document.querySelector('.notification');
    if (existing) {
        existing.remove();
    }

    // Create notification
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 'fa-check-circle'}"></i>
        <span>${message}</span>
    `;

    // Add styles
    const bgColor = type === 'error' ? 'rgba(239, 68, 68, 0.95)' : 'rgba(16, 185, 129, 0.95)';
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 24px;
        background: ${bgColor};
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 9999;
        animation: slideIn 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        font-family: 'Outfit', sans-serif;
    `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

/**
 * Initialize smooth scrolling
 */
function initializeSmoothScroll() {
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
 * Start new assessment
 */
function newAssessment() {
    const form = document.getElementById('injuryForm');
    const resultsSection = document.getElementById('results');
    
    if (form) {
        form.reset();
    }
    
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }

    // Reset gauges
    const quickGauge = document.getElementById('quickGauge');
    const extendedGauge = document.getElementById('extendedGauge');
    if (quickGauge) quickGauge.style.strokeDashoffset = '283';
    if (extendedGauge) extendedGauge.style.strokeDashoffset = '283';

    // Scroll to form
    const formSection = document.getElementById('prediction-form');
    if (formSection) {
        formSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Add CSS animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);
