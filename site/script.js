/* ============================================================
   agentmemory.md — Landing page interactions
   ============================================================ */

/* ── Hero word rotation ────────────────────────────────────── */
(function () {
  const words = ['Cursor', 'Claude', 'ChatGPT', 'OpenClaw'];
  const el = document.getElementById('rotating-word');
  if (!el) return;

  let current = 0;

  function rotate() {
    // Fade out
    el.classList.add('fade-out');
    el.classList.remove('fade-in-word');

    setTimeout(() => {
      current = (current + 1) % words.length;
      el.textContent = words[current];

      // Fade in
      el.classList.remove('fade-out');
      el.classList.add('fade-in-word');
    }, 320); // matches CSS transition duration
  }

  // Start after a short delay, then repeat every 2.2s
  setTimeout(() => {
    rotate();
    setInterval(rotate, 2200);
  }, 1800);
})();

/* ── Scroll fade-in ────────────────────────────────────────── */
(function () {
  const targets = document.querySelectorAll('.fade-in');
  if (!targets.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.1,
      rootMargin: '0px 0px -32px 0px',
    }
  );

  targets.forEach((el) => observer.observe(el));
})();

/* ── Smooth nav highlight on scroll ───────────────────────── */
(function () {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  if (!sections.length || !navLinks.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          navLinks.forEach((link) => {
            link.style.color = '';
            if (link.getAttribute('href') === '#' + id) {
              link.style.color = 'var(--text)';
            }
          });
        }
      });
    },
    { threshold: 0.4 }
  );

  sections.forEach((s) => observer.observe(s));
})();
