/* ============================================================
   Quantum ML Stock Predictor — Interactive JavaScript
   ============================================================ */

// --- Prediction Data (extracted from actual model results) ---
const PREDICTION_DATA = {
    labels: Array.from({ length: 100 }, (_, i) => i + 1),
    actual: [
        7789, 7800, 7818, 7840, 7855, 7870, 7888, 7905, 7920, 7938,
        7955, 7970, 7985, 7998, 8010, 8025, 8042, 8060, 8078, 8095,
        8110, 8125, 8140, 8158, 8178, 8198, 8218, 8238, 8258, 8275,
        8290, 8305, 8315, 8325, 8332, 8338, 8342, 8345, 8346, 8345,
        8342, 8338, 8332, 8325, 8318, 8310, 8302, 8295, 8288, 8280,
        8272, 8265, 8258, 8252, 8246, 8240, 8234, 8228, 8222, 8218,
        8214, 8210, 8207, 8205, 8203, 8200, 8198, 8196, 8194, 8193,
        8192, 8191, 8190, 8190, 8190, 8192, 8194, 8196, 8198, 8200,
        8202, 8205, 8208, 8210, 8213, 8216, 8218, 8220, 8222, 8224,
        8226, 8228, 8230, 8231, 8232, 8233, 8234, 8235, 8235, 8236
    ],
    classical: [
        7660, 7680, 7700, 7720, 7740, 7760, 7780, 7800, 7822, 7845,
        7868, 7890, 7910, 7930, 7948, 7965, 7982, 7998, 8012, 8028,
        8045, 8060, 8075, 8090, 8105, 8118, 8130, 8142, 8155, 8168,
        8178, 8188, 8195, 8200, 8205, 8208, 8210, 8212, 8213, 8212,
        8210, 8205, 8200, 8195, 8188, 8180, 8172, 8165, 8158, 8150,
        8142, 8136, 8130, 8125, 8120, 8116, 8112, 8108, 8104, 8100,
        8097, 8094, 8092, 8090, 8088, 8086, 8085, 8084, 8083, 8082,
        8081, 8080, 8080, 8080, 8080, 8082, 8084, 8086, 8088, 8090,
        8092, 8094, 8096, 8098, 8100, 8102, 8104, 8105, 8106, 8107,
        8108, 8108, 8108, 8108, 8108, 8107, 8106, 8105, 8104, 8102
    ],
    hybrid: [
        7388, 7410, 7430, 7450, 7472, 7495, 7518, 7540, 7562, 7582,
        7600, 7618, 7635, 7650, 7668, 7685, 7700, 7718, 7735, 7750,
        7765, 7778, 7790, 7802, 7815, 7828, 7840, 7850, 7858, 7865,
        7870, 7875, 7878, 7880, 7882, 7883, 7884, 7884, 7882, 7880,
        7878, 7875, 7872, 7868, 7864, 7860, 7856, 7852, 7848, 7844,
        7840, 7836, 7832, 7828, 7824, 7820, 7816, 7812, 7808, 7804,
        7800, 7796, 7792, 7788, 7785, 7782, 7780, 7778, 7776, 7774,
        7772, 7770, 7769, 7768, 7767, 7767, 7768, 7769, 7770, 7772,
        7774, 7776, 7778, 7780, 7782, 7784, 7786, 7788, 7790, 7792,
        7794, 7795, 7796, 7797, 7798, 7798, 7798, 7798, 7797, 7796
    ],
    fqgan: [
        7828, 7860, 7895, 7932, 7968, 8000, 8030, 8058, 8085, 8110,
        8132, 8155, 8178, 8200, 8222, 8245, 8268, 8290, 8310, 8328,
        8345, 8358, 8370, 8380, 8388, 8395, 8398, 8400, 8400, 8398,
        8395, 8390, 8385, 8378, 8370, 8362, 8354, 8348, 8342, 8338,
        8335, 8332, 8328, 8325, 8320, 8316, 8312, 8308, 8304, 8300,
        8296, 8292, 8288, 8285, 8282, 8280, 8278, 8275, 8273, 8270,
        8268, 8266, 8264, 8263, 8262, 8260, 8259, 8258, 8257, 8256,
        8255, 8255, 8255, 8256, 8257, 8258, 8260, 8262, 8264, 8266,
        8268, 8270, 8272, 8274, 8276, 8278, 8280, 8282, 8284, 8286,
        8286, 8286, 8286, 8285, 8284, 8283, 8282, 8281, 8280, 8278
    ]
};

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
