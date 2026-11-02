import { useEffect, useRef } from "react";

export default function AnimatedBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;

    function resize() {
      if (!canvas) return;
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    const NAVY    = "rgba(36,54,96,";
    const CRIMSON = "rgba(124,24,48,";
    const GREEN   = "rgba(22,163,74,";
    const GRID    = 44;
    const NODE_COUNT = 32;

    const nodes = Array.from({ length: NODE_COUNT }, () => ({
      x:     Math.random() * canvas.width,
      y:     Math.random() * canvas.height,
      vx:    (Math.random() - 0.5) * 0.16,
      vy:    (Math.random() - 0.5) * 0.16,
      r:     Math.random() * 1.2 + 0.4,
      pulse: Math.random() * Math.PI * 2,
    }));

    const blips: { x: number; y: number; life: number }[] = [];
    let sweep = 0;

    function draw() {
      if (!canvas || !ctx) return;
      const W = canvas.width, H = canvas.height;
      const CX = W * 0.75, CY = H * 0.42, R = Math.min(W, H) * 0.22;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0a0c0f";
      ctx.fillRect(0, 0, W, H);

      // Grid
      ctx.strokeStyle = NAVY + "0.15)";
      ctx.lineWidth = 0.5;
      for (let x = 0; x < W; x += GRID) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
      for (let y = 0; y < H; y += GRID) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      }

      // Radar rings
      ctx.strokeStyle = NAVY + "0.1)";
      ctx.lineWidth = 0.4;
      for (let d = 1; d <= 5; d++) {
        ctx.beginPath();
        ctx.arc(CX, CY, R * d / 3, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Radar spokes
      ctx.strokeStyle = NAVY + "0.08)";
      for (let a = 0; a < 8; a++) {
        const ang = (a / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(CX, CY);
        ctx.lineTo(CX + Math.cos(ang) * R * 1.8, CY + Math.sin(ang) * R * 1.8);
        ctx.stroke();
      }

      // Sweep cone
      ctx.save();
      ctx.translate(CX, CY);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.arc(0, 0, R * 1.4, sweep - 0.65, sweep);
      ctx.closePath();
      const sg = ctx.createLinearGradient(0, 0,
        Math.cos(sweep) * R, Math.sin(sweep) * R);
      sg.addColorStop(0,   CRIMSON + "0)");
      sg.addColorStop(0.6, CRIMSON + "0)");
      sg.addColorStop(1,   CRIMSON + "0.28)");
      ctx.fillStyle = sg;
      ctx.fill();
      ctx.restore();

      // Sweep line
      ctx.strokeStyle = CRIMSON + "0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(CX, CY);
      ctx.lineTo(CX + Math.cos(sweep) * R * 1.4, CY + Math.sin(sweep) * R * 1.4);
      ctx.stroke();
      sweep += 0.011;

      // Blips
      for (let i = blips.length - 1; i >= 0; i--) {
        const b = blips[i];
        b.life -= 0.01;
        if (b.life <= 0) { blips.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(b.x, b.y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle   = GREEN + (b.life * 0.75) + ")";
        ctx.fill();
        ctx.strokeStyle = GREEN + (b.life * 0.4) + ")";
        ctx.lineWidth   = 0.5;
        ctx.stroke();
      }
      if (Math.random() < 0.022) {
        const ang = Math.random() * Math.PI * 2;
        const dist = Math.random() * R;
        blips.push({ x: CX + Math.cos(ang) * dist, y: CY + Math.sin(ang) * dist, life: 1 });
      }

      // Drifting nodes
      nodes.forEach(n => {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > W) n.vx *= -1;
        if (n.y < 0 || n.y > H) n.vy *= -1;
        n.pulse += 0.014;
        const glow = (Math.sin(n.pulse) + 1) * 0.5;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + glow * 0.8, 0, Math.PI * 2);
        ctx.fillStyle = NAVY + (0.28 + glow * 0.38) + ")";
        ctx.fill();
      });

      // Node connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            ctx.strokeStyle = NAVY + ((1 - dist / 100) * 0.1) + ")";
            ctx.lineWidth   = 0.5;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      // Horizontal scan line
      const scanY = ((Date.now() * 0.03) % (H + 20)) - 10;
      const scanG = ctx.createLinearGradient(0, scanY - 6, 0, scanY + 6);
      scanG.addColorStop(0,   "rgba(36,54,96,0)");
      scanG.addColorStop(0.5, "rgba(36,54,96,0.05)");
      scanG.addColorStop(1,   "rgba(36,54,96,0)");
      ctx.fillStyle = scanG;
      ctx.fillRect(0, scanY - 6, W, 12);

      animId = requestAnimationFrame(draw);
    }

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  );
}
