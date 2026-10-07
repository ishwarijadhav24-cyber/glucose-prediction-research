"use client";

import { useEffect, useRef } from "react";

/**
 * Interactive canvas background: drifting particles joined by thin links (a "sensor network").
 * Particles are pushed away from the cursor and draw links to it; a click sends out a ripple pulse.
 * Purely decorative: pointer-events are off, it pauses when the tab is hidden and is disabled for
 * users who prefer reduced motion.
 */
export function InteractiveBackground() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    type P = { x: number; y: number; vx: number; vy: number; r: number };
    type Ripple = { x: number; y: number; t: number };
    let w = 0, h = 0, dpr = 1, raf = 0;
    let particles: P[] = [];
    const ripples: Ripple[] = [];
    const mouse = { x: -9999, y: -9999, active: false };
    let rgb = "59,130,246";
    let dark = false;

    const readColor = () => {
      dark = document.documentElement.classList.contains("dark");
      const v = getComputedStyle(document.documentElement).getPropertyValue("--primary").trim(); // "221.2 83.2% 53.3%"
      const [hh, ss, ll] = v.split(/\s+/).map((x) => parseFloat(x));
      if ([hh, ss, ll].every((n) => Number.isFinite(n))) {
        const s = ss / 100, l = ll / 100, k = (n: number) => (n + hh / 30) % 12;
        const a = s * Math.min(l, 1 - l);
        const f = (n: number) => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))));
        rgb = `${f(0)},${f(8)},${f(4)}`;
      }
    };

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.round(Math.min(120, (w * h) / 14000));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35,
        r: 1 + Math.random() * 1.6,
      }));
    };

    const LINK = 130, PUSH = 140;
    const frame = () => {
      ctx.clearRect(0, 0, w, h);
      const base = dark ? 0.55 : 0.4;

      for (const rp of ripples) rp.t += 1;
      while (ripples.length && ripples[0].t > 90) ripples.shift();

      for (const p of particles) {
        if (mouse.active) {
          const dx = p.x - mouse.x, dy = p.y - mouse.y, d = Math.hypot(dx, dy);
          if (d < PUSH && d > 0.1) { const f = ((PUSH - d) / PUSH) * 0.6; p.vx += (dx / d) * f; p.vy += (dy / d) * f; }
        }
        for (const rp of ripples) {
          const dx = p.x - rp.x, dy = p.y - rp.y, d = Math.hypot(dx, dy), front = rp.t * 6;
          if (Math.abs(d - front) < 22 && d > 0.1) { const f = 1.6 * (1 - rp.t / 90); p.vx += (dx / d) * f; p.vy += (dy / d) * f; }
        }
        p.vx *= 0.96; p.vy *= 0.96;
        p.vx += (Math.random() - 0.5) * 0.02; p.vy += (Math.random() - 0.5) * 0.02;
        const sp = Math.hypot(p.vx, p.vy);
        if (sp < 0.12) { p.vx += (Math.random() - 0.5) * 0.08; p.vy += (Math.random() - 0.5) * 0.08; }
        p.x += p.vx; p.y += p.vy;
        if (p.x < -10) p.x = w + 10; if (p.x > w + 10) p.x = -10;
        if (p.y < -10) p.y = h + 10; if (p.y > h + 10) p.y = -10;
      }

      ctx.lineWidth = 1;
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i];
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j], d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < LINK) {
            ctx.strokeStyle = `rgba(${rgb},${(1 - d / LINK) * base * 0.45})`;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
        if (mouse.active) {
          const d = Math.hypot(a.x - mouse.x, a.y - mouse.y);
          if (d < LINK * 1.5) {
            ctx.strokeStyle = `rgba(${rgb},${(1 - d / (LINK * 1.5)) * base * 0.9})`;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
          }
        }
      }
      for (const p of particles) {
        ctx.fillStyle = `rgba(${rgb},${base})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      }
      for (const rp of ripples) {
        ctx.strokeStyle = `rgba(${rgb},${0.35 * (1 - rp.t / 90)})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(rp.x, rp.y, rp.t * 6, 0, Math.PI * 2); ctx.stroke();
      }
      raf = requestAnimationFrame(frame);
    };

    const onMove = (e: PointerEvent) => { mouse.x = e.clientX; mouse.y = e.clientY; mouse.active = true; };
    const onLeave = () => { mouse.active = false; };
    const onDown = (e: PointerEvent) => { ripples.push({ x: e.clientX, y: e.clientY, t: 0 }); };
    const onVis = () => {
      cancelAnimationFrame(raf);
      if (!document.hidden && !reduce) raf = requestAnimationFrame(frame);
    };

    readColor();
    resize();
    const obs = new MutationObserver(readColor);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });
    document.addEventListener("pointerleave", onLeave);
    document.addEventListener("visibilitychange", onVis);
    if (reduce) frame(); // draw one static frame
    else raf = requestAnimationFrame(frame);
    if (reduce) cancelAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(raf);
      obs.disconnect();
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerdown", onDown);
      document.removeEventListener("pointerleave", onLeave);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return <canvas ref={ref} aria-hidden className="pointer-events-none fixed inset-0 z-0" />;
}
