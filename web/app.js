/* ============================================================
   Quantum ML Stock Predictor — Interactive JavaScript
   ============================================================ */

// --- Prediction Data (extracted from actual model results) ---
// PREDICTION_DATA is loaded from data.js


const INSIGHTS = {
    classical: 'The Classical GAN (LSTM-based) tracks the overall FTSE trend well, maintaining a consistent offset below the actual prices. Its 50-epoch training captures sequential patterns effectively.',
    hybrid: 'The Hybrid Quantum GAN shows a larger gap from actual prices — expected with only 1 training epoch on quantum hardware. The quantum circuit generator captures the trend direction but with reduced magnitude.',
    fqgan: 'The Fully Quantum GAN achieves the closest match, slightly overshooting peaks. Its amplitude encoding and SWAP-test discrimination capture both trend direction and magnitude remarkably well.',
    all: 'Comparing all three architectures: FQGAN leads in accuracy, Classical GAN provides reliable tracking, while the Hybrid QGAN captures directional trends but with lower magnitude fidelity.'
};

// --- Particle Background ---
function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    let animFrameId;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createParticles() {
        particles = [];
        const count = Math.min(Math.floor(window.innerWidth / 15), 80);
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 1.5 + 0.5,
                speedX: (Math.random() - 0.5) * 0.3,
                speedY: (Math.random() - 0.5) * 0.3,
                opacity: Math.random() * 0.5 + 0.1,
                hue: Math.random() > 0.5 ? 258 : 190 // purple or cyan
            });
        }
    }

    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach((p, i) => {
            p.x += p.speedX;
            p.y += p.speedY;

            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${p.hue}, 80%, 65%, ${p.opacity})`;
            ctx.fill();

            // Draw connections
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[j].x - p.x;
                const dy = particles[j].y - p.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `hsla(258, 80%, 65%, ${0.08 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        });

        animFrameId = requestAnimationFrame(drawParticles);
    }

    resize();
    createParticles();
    drawParticles();

    window.addEventListener('resize', () => {
        resize();
        createParticles();
    });
}

// --- Navbar Scroll ---
function initNavbar() {
    const navbar = document.getElementById('navbar');
    const toggle = document.getElementById('nav-toggle');
    const links = document.querySelector('.nav-links');

    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
    });

    if (toggle && links) {
        toggle.addEventListener('click', () => {
            links.classList.toggle('open');
        });

        // Close mobile menu on link click
        links.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                links.classList.remove('open');
            });
        });
    }
}

// --- Chart ---
let predictionChart = null;

function getChartConfig(type) {
    const data = PREDICTION_DATA;
    const datasets = [];
    
    // Always include actual
    datasets.push({
        label: 'Actual FTSE',
        data: data.actual,
        borderColor: '#e8e8f0',
        backgroundColor: 'rgba(232, 232, 240, 0.05)',
        borderWidth: 2.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.4,
        fill: false,
        order: 0
    });

    const modelStyles = {
        classical: { label: 'Classical GAN', color: '#8b5cf6', data: data.classical },
        hybrid: { label: 'Hybrid QGAN', color: '#06b6d4', data: data.hybrid },
        fqgan: { label: 'Fully Quantum GAN', color: '#ec4899', data: data.fqgan }
    };

    if (type === 'all') {
        Object.values(modelStyles).forEach(m => {
            datasets.push({
                label: m.label,
                data: m.data,
                borderColor: m.color,
                backgroundColor: `${m.color}08`,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.4,
                fill: false,
                borderDash: [],
                order: 1
            });
        });
    } else {
        const m = modelStyles[type];
        datasets.push({
            label: m.label,
            data: m.data,
            borderColor: m.color,
            backgroundColor: `${m.color}15`,
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.4,
            fill: true,
            order: 1
        });
    }

    return {
        type: 'line',
        data: { labels: data.labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800, easing: 'easeInOutQuart' },
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: '#8888a8',
                        font: { family: "'Inter', sans-serif", size: 12, weight: '500' },
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 10, 15, 0.9)',
                    titleColor: '#e8e8f0',
                    bodyColor: '#8888a8',
                    borderColor: 'rgba(139, 92, 246, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: { family: "'Inter', sans-serif", size: 13, weight: '600' },
                    bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
                    callbacks: {
                        title: (items) => `Sample ${items[0].label}`,
                        label: (item) => ` ${item.dataset.label}: ${item.parsed.y.toLocaleString()}`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Test Sample Index', color: '#555570', font: { size: 12, weight: '500' } },
                    ticks: { color: '#555570', font: { size: 11 }, maxTicksLimit: 10 },
                    grid: { color: 'rgba(255, 255, 255, 0.03)', drawBorder: false }
                },
                y: {
                    title: { display: true, text: 'FTSE Price', color: '#555570', font: { size: 12, weight: '500' } },
                    ticks: {
                        color: '#555570',
                        font: { size: 11 },
                        callback: (v) => v.toLocaleString()
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.03)', drawBorder: false }
                }
            }
        }
    };
}

function initChart() {
    const ctx = document.getElementById('prediction-chart');
    if (!ctx) return;

    ctx.parentElement.style.height = '400px';
    predictionChart = new Chart(ctx, getChartConfig('classical'));

    // Tab switching
    document.querySelectorAll('.chart-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const type = tab.dataset.chart;
            predictionChart.destroy();
            predictionChart = new Chart(ctx, getChartConfig(type));

            document.getElementById('insight-text').textContent = INSIGHTS[type];
        });
    });
}

// --- Scroll Reveal ---
function initScrollReveal() {
    const elements = document.querySelectorAll('.model-card, .metric-card, .tech-item, .pipeline-step, .section-header, .chart-container, .chart-insight, .metrics-summary, .summary-card');
    
    elements.forEach(el => el.classList.add('reveal'));

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, index * 80);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    elements.forEach(el => observer.observe(el));
}

// --- Metric Bar Animation ---
function initMetricBars() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.querySelectorAll('.bar-fill').forEach((bar, i) => {
                    setTimeout(() => bar.classList.add('animated'), i * 200);
                });
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    document.querySelectorAll('.metric-card').forEach(card => observer.observe(card));
}

// --- Smooth Scroll for Anchor Links ---
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// --- Initialize Everything ---
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initNavbar();
    initChart();
    initScrollReveal();
    initMetricBars();
    initSmoothScroll();
});
