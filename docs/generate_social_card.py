"""Generate a 1280x640 social preview card for helix-rag."""

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640

BG        = "#ffffff"
DARK      = "#0f172a"
MID       = "#334155"
DIM       = "#64748b"
SEP       = "#e2e8f0"
ARROW     = "#64748b"

# Section block backgrounds + borders
IDX_BLOCK = "#eff6ff"; IDX_BORDER = "#93c5fd"
QRY_BLOCK = "#f5f3ff"; QRY_BORDER = "#c4b5fd"
AGT_BLOCK = "#f8fafc"; AGT_BORDER = "#475569"

# Node styles: white fill, colored border, dark text
N_BLUE_BD = "#2563eb"; N_BLUE_TX = "#1d4ed8"
N_PUR_BD  = "#7c3aed"; N_PUR_TX  = "#5b21b6"
N_GRN_BD  = "#16a34a"; N_GRN_TX  = "#15803d"
N_FILL    = "#ffffff"


def rr(draw, x, y, w, h, r, fill, outline, lw=2):
    draw.rounded_rectangle([x, y, x+w, y+h], radius=r,
                           fill=fill, outline=outline, width=lw)


def arr_h(draw, x1, y, x2):
    draw.line([(x1, y), (x2, y)], fill=ARROW, width=2)
    draw.polygon([(x2, y), (x2-9, y-5), (x2-9, y+5)], fill=ARROW)


def arr_v(draw, x, y1, y2):
    draw.line([(x, y1), (x, y2)], fill=ARROW, width=2)
    draw.polygon([(x, y2), (x-5, y2-9), (x+5, y2-9)], fill=ARROW)


def text_c(draw, cx, cy, t, font, color):
    bb = draw.textbbox((0, 0), t, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.text((cx - tw//2, cy - th//2), t, font=font, fill=color)


def load(size, bold=False):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


img  = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

f_title  = load(54, bold=True)
f_sub    = load(20)
f_tag    = load(19)
f_sec    = load(13, bold=True)
f_label  = load(17, bold=True)
f_small  = load(14)
f_agent  = load(16, bold=True)
f_agtsub = load(12)

PAD   = 56   # left/right margin
BOX_W = 158
BOX_H = 82
GAP   = 24

# ── Header ────────────────────────────────────────────────────────────────
draw.text((PAD, 32), "helix-rag", font=f_title, fill=DARK)
draw.text((PAD+4, 96), "Agentic search across any document collection. Every answer grounded and cited.",
          font=f_sub, fill=MID)
draw.line([(PAD, 128), (W-PAD, 128)], fill=SEP, width=1)

# ── Section block helper ───────────────────────────────────────────────────
BLOCK_PAD = 14   # padding inside block around nodes
BLOCK_X   = PAD - BLOCK_PAD

def section_block(y_top, label, label_color, block_fill, block_border,
                  steps, node_bd, node_tx):
    n = len(steps)
    total_nodes = n * BOX_W + (n-1) * GAP
    nodes_x = (W - total_nodes) // 2
    LABEL_H = 28
    block_w = W - 2*(PAD - BLOCK_PAD)
    block_h = LABEL_H + BLOCK_PAD + BOX_H + BLOCK_PAD

    # block background
    rr(draw, BLOCK_X, y_top, block_w, block_h, 12,
       block_fill, block_border, lw=2)

    # section label inside top-left of block
    draw.text((BLOCK_X + 14, y_top + 7), label, font=f_sec, fill=label_color)

    nodes_y = y_top + LABEL_H + BLOCK_PAD
    centers = []
    for i, (lbl, _) in enumerate(steps):
        x = nodes_x + i*(BOX_W+GAP)
        rr(draw, x, nodes_y, BOX_W, BOX_H, 9, N_FILL, node_bd, lw=2)
        lines = lbl.split("\n")
        lh = 21
        for j, ln in enumerate(lines):
            cy = nodes_y + BOX_H//2 - (len(lines)*lh)//2 + j*lh + lh//2
            text_c(draw, x + BOX_W//2, cy, ln, f_label, node_tx)
        if i < n-1:
            x1 = x + BOX_W + 3; x2 = x + BOX_W + GAP - 3
            arr_h(draw, x1, nodes_y + BOX_H//2, x2)
        centers.append(x + BOX_W//2)

    return nodes_y, centers, block_h


# ── Row 1: Indexing ───────────────────────────────────────────────────────
r1_steps = [
    ("PDF Files",             None),
    ("PyMuPDF\nExtraction",   None),
    ("Parent-Child\nChunker", None),
    ("OpenAI\nEmbeddings",    None),
    ("Qdrant\nVector Store",  None),
]
R1_Y = 140
r1_nodes_y, r1_centers, r1_block_h = section_block(
    R1_Y, "INDEXING", N_BLUE_TX, IDX_BLOCK, IDX_BORDER,
    r1_steps, N_BLUE_BD, N_BLUE_TX)

qdrant_cx = r1_centers[-1]
qdrant_bottom = r1_nodes_y + BOX_H

# ── Agent box ─────────────────────────────────────────────────────────────
AGT_W = 240; AGT_H = 52
AGT_X = (W - AGT_W) // 2
AGT_Y = R1_Y + r1_block_h + 30

arr_v(draw, qdrant_cx, qdrant_bottom + 4, AGT_Y - 4)

rr(draw, AGT_X, AGT_Y, AGT_W, AGT_H, 10, AGT_BLOCK, AGT_BORDER, lw=2)
text_c(draw, AGT_X + AGT_W//2, AGT_Y + 16, "Agent Layer", f_agent, DARK)
text_c(draw, AGT_X + AGT_W//2, AGT_Y + 36, "routes questions · returns answers", f_agtsub, DIM)

# ── Row 2: Querying ───────────────────────────────────────────────────────
r2_steps = [
    ("User Question",         None),
    ("Input Guardrails",      None),
    ("Hybrid Search\nBM25 + Dense", None),
    ("Cross-Encoder\nReranker", None),
    ("GPT-4o\nGrounded Gen",  None),
    ("Answer + Sources",      None),
]

# mixed colors: green for endpoints, blue for retrieval, purple for LLM/guardrails
r2_colors = [
    (N_GRN_BD, N_GRN_TX),
    (N_PUR_BD, N_PUR_TX),
    (N_BLUE_BD, N_BLUE_TX),
    (N_BLUE_BD, N_BLUE_TX),
    (N_PUR_BD, N_PUR_TX),
    (N_GRN_BD, N_GRN_TX),
]

R2_Y = AGT_Y + AGT_H + 30
LABEL_H2 = 28; BLOCK_PAD2 = 14
block_w = W - 2*(PAD - BLOCK_PAD)
block_h2 = LABEL_H2 + BLOCK_PAD2 + BOX_H + BLOCK_PAD2
rr(draw, BLOCK_X, R2_Y, block_w, block_h2, 12, QRY_BLOCK, QRY_BORDER, lw=2)
draw.text((BLOCK_X + 14, R2_Y + 7), "QUERYING", font=f_sec, fill=N_PUR_TX)

n2 = len(r2_steps)
total2 = n2*BOX_W + (n2-1)*GAP
r2_nodes_x = (W - total2) // 2
r2_nodes_y = R2_Y + LABEL_H2 + BLOCK_PAD2

# Arrow from agent down into querying block (centered under agent box)
arr_v(draw, AGT_X + AGT_W//2, AGT_Y + AGT_H + 4, R2_Y - 4)

for i, (lbl, _) in enumerate(r2_steps):
    bd, tx = r2_colors[i]
    x = r2_nodes_x + i*(BOX_W+GAP)
    rr(draw, x, r2_nodes_y, BOX_W, BOX_H, 9, N_FILL, bd, lw=2)
    lines = lbl.split("\n")
    lh = 21
    for j, ln in enumerate(lines):
        cy = r2_nodes_y + BOX_H//2 - (len(lines)*lh)//2 + j*lh + lh//2
        text_c(draw, x + BOX_W//2, cy, ln, f_label, tx)
    if i < n2-1:
        x1 = x + BOX_W + 3; x2 = x + BOX_W + GAP - 3
        arr_h(draw, x1, r2_nodes_y + BOX_H//2, x2)

# ── Footer badges ─────────────────────────────────────────────────────────
draw.line([(PAD, R2_Y + block_h2 + 18), (W-PAD, R2_Y + block_h2 + 18)],
          fill=SEP, width=1)

badges = [
    ("Hybrid BM25 + Dense",    IDX_BLOCK, N_BLUE_BD, N_BLUE_TX),
    ("Evaluation + Guardrails",AGT_BLOCK, AGT_BORDER, DARK),
    ("Agentic RAG",            QRY_BLOCK, N_PUR_BD,  N_PUR_TX),
    ("CI Quality Gate",        "#f0fdf4", N_GRN_BD,  N_GRN_TX),
]
BDG_W = 214; BDG_H = 32; BDG_GAP = 16
BDG_TOTAL = len(badges)*BDG_W + (len(badges)-1)*BDG_GAP
bdg_x = (W - BDG_TOTAL) // 2
bdg_y = R2_Y + block_h2 + 26

for label, bg, bd, tx in badges:
    rr(draw, bdg_x, bdg_y, BDG_W, BDG_H, 6, bg, bd, lw=1)
    text_c(draw, bdg_x + BDG_W//2, bdg_y + BDG_H//2, label, f_small, tx)
    bdg_x += BDG_W + BDG_GAP

img.save("docs/social_preview.png", "PNG")
total_h = R2_Y + block_h2 + 26 + BDG_H
print(f"Content bottom: {total_h}px  Canvas: {H}px  Remaining: {H-total_h}px")
