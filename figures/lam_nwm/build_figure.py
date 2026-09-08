#!/usr/bin/env python3
"""Build editable SVGs, vector PDFs and PNG previews for the LAM / NWM paper.

Requires cairosvg. All artwork is native vector geometry; no external assets.
Run: python build_figure.py
"""
from pathlib import Path
import re
from html import escape
from functools import lru_cache

import cairosvg
from PIL import ImageFont

OUT = Path(__file__).resolve().parent
W, H = 1600, 1984
C = {
    'ink': '#202B3A', 'muted': '#566475', 'line': '#64748B',
    'border': '#D4DDE5', 'paper': '#FFFFFF', 'neutral': '#F6F8FA',
    'purple': '#7755A6', 'purple_bg': '#F2EDF8',
    'blue': '#326DA8', 'blue_bg': '#ECF3FA',
    'teal': '#267F79', 'teal_bg': '#EAF5F2',
    'orange': '#B3652D', 'orange_bg': '#FFF1E5',
    'frozen': '#EEF1F5',
}
parts = []


def add(s):
    parts.append(s)


def rect(x, y, w, h, fill='white', stroke=None, radius=10, sw=1.5, dash=None):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"'
        + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')


@lru_cache(None)
def font(size, weight=400, italic=False):
    style = ('BoldItalic' if italic else 'Bold') if weight >= 600 else ('Italic' if italic else 'Regular')
    root = Path('/usr/share/fonts/truetype/liberation')
    return ImageFont.truetype(str(root/f'LiberationSans-{style}.ttf'), round(size*10))


def length(s, size, weight=400, italic=False):
    return font(size,weight,italic).getlength(s)/10


def runs(s,size,weight,italic):
    result=[]
    last=0
    for m in re.finditer(r'\{([^{}]+)_([^{}]+)\}', s):
        result.append((s[last:m.start()],size,0,italic))
        base, sub = m.groups()
        result.extend([(base,size,0,True),(sub,size*.69,size*.22,False)])
        last=m.end()
    result.append((s[last:],size,0,italic))
    return result


def text(x, y, s, size=24, color=None, weight=400, anchor='start', italic=False):
    # Position math runs explicitly: avoids SVG renderer inconsistencies with
    # nested tspan anchors and missing combining-accent glyphs in PDF fonts.
    segments=runs(s,size,weight,italic)
    cleaned=lambda q:q.replace('x̂','x').replace('f̂','f').replace('ẑ','z').replace('ε̂','ε').replace('Σ̂','Σ').replace('û','u')
    widths=[length(cleaned(q),z,weight,it) for q,z,dy,it in segments]
    total=sum(widths)
    cursor=x-(total/2 if anchor=='middle' else total if anchor=='end' else 0)
    for (q,z,dy,it),width in zip(segments,widths):
        content=cleaned(q)
        add(f'<text xml:space="preserve" x="{cursor:.3f}" y="{y+dy:.3f}" font-family="Liberation Sans, Arial, sans-serif" '
            f'font-size="{z}" font-weight="{weight}" fill="{color or C["ink"]}"'
            + (' font-style="italic"' if it else '') + f'>{escape(content)}</text>')
        if q in ['x̂','f̂','ẑ','ε̂','Σ̂','û']:
            path([(cursor+width*.17,y+dy-z*.80),(cursor+width*.57,y+dy-z*.97),
                  (cursor+width*.98,y+dy-z*.80)],color=color or C['ink'],width=1.5)
        cursor+=width


def path(points, color=None, width=2, arrow=False, dash=None):
    d = 'M ' + ' L '.join(f'{x},{y}' for x, y in points)
    color = color or C['line']
    marker = 'purple' if color == C['purple'] else 'default'
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"'
        + (f' marker-end="url(#{marker})"' if arrow else '')
        + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')


def arrow(x1, y1, x2, y2, **kwargs):
    path([(x1, y1), (x2, y2)], arrow=True, **kwargs)


def lock(x, y):
    add(f'<path d="M {x+3},{y+7} v-3 a4,4 0 0 1 8,0 v3" fill="none" '
        f'stroke="{C["muted"]}" stroke-width="1.5"/>')
    rect(x, y+6, 14, 11, C['frozen'], C['muted'], 2, 1.4)


def block(x, y, w, h, label, kind='purple', frozen=False, size=25, subtitle=None):
    rect(x, y, w, h, C['frozen'] if frozen else C[kind+'_bg'],
         C['muted'] if frozen else C[kind], 8, 1.8, '6 4' if frozen else None)
    text(x+w/2, y+h/2+(1 if subtitle else 8), label, size=size, anchor='middle')
    if subtitle:
        text(x+w/2, y+h/2+24, subtitle, size=20, color=C['muted'], anchor='middle')
    if frozen:
        lock(x+w-19, y+4)


def scene(x, y, w=43, h=31, future=False):
    """A small vector observation icon, intentionally not a dataset sample."""
    rect(x, y, w, h, '#EDF3F6', '#9FB0BC', 3, 1)
    add(f'<path d="M{x+1},{y+1} L{x+w*.43},{y+h*.42} L{x+w*.43},{y+h*.67} '
        f'L{x+1},{y+h-1} Z" fill="#C8DCD8"/>')
    add(f'<path d="M{x+w-1},{y+1} L{x+w*.64},{y+h*.42} L{x+w*.64},{y+h*.67} '
        f'L{x+w-1},{y+h-1} Z" fill="#C9D7E4"/>')
    add(f'<path d="M{x+1},{y+h-1} L{x+w*.43},{y+h*.62} L{x+w*.64},{y+h*.62} '
        f'L{x+w-1},{y+h-1} Z" fill="#DFD9CD"/>')
    shift = 4 if future else 0
    rect(x+w*.45-shift, y+h*.29, w*.15+shift, h*.32, '#A9B8C4', radius=0)


def pair(x, cy, compact=False):
    w, gap = (33, 7) if compact else (43, 16)
    scene(x, cy-22, w, 31)
    scene(x+w+gap, cy-22, w, 31, True)
    text(x+w/2, cy+35, '{x_t}', 21, anchor='middle')
    text(x+1.5*w+gap, cy+35, '{x_t+Δt}', 21, anchor='middle')
    return x+2*w+gap


def feature(x, y, w=30):
    for i in range(3):
        for j in range(3):
            shades = ['#C6B4DD', '#E2D7EF', '#AABCD8', '#C4DCDB']
            rect(x+j*w/3, y+i*w/3, w/3-1, w/3-1, shades[(i+j*2)%4], radius=1)


def section(y, letter, title, note=None):
    text(24, y, f'({letter})', 30, weight=700)
    text(78, y, title, 30, weight=700)
    if note:
        text(1576, y, note, 23, color=C['muted'], anchor='end')


def card(x, y, w, h, title, kind, tag):
    rect(x, y, w, h, '#FFFFFF', C['border'], 12)
    rect(x, y, 6, h, C[kind], radius=3)
    text(x+21, y+33, title, 25, weight=700, color=C[kind])
    text(x+w-19, y+33, tag, 22, color=C['muted'], anchor='end')


def loss(x, y, s, kind='muted', anchor='middle', size=23):
    text(x, y, s, size=size, color=C[kind], anchor=anchor)


def panel_a():
    section(37, 'a', 'Latent action models', 'Which reconstruction target yields transferable actions?')
    # Action only.
    x, y = 24, 67
    card(x, y, 768, 183, 'Action', 'orange', 'GT action supervision')
    cy = 153
    end = pair(x+24, cy)
    arrow(end+12, cy, x+192, cy)
    block(x+198, cy-28, 118, 56, '{E_LAM}')
    arrow(x+322, cy, x+363, cy)
    text(x+384, cy+8, 'z', 29, C['purple'], anchor='middle', italic=True)
    arrow(x+406, cy, x+450, cy)
    block(x+456, cy-28, 123, 56, '{D_a}', 'orange')
    arrow(x+585, cy, x+636, cy)
    text(x+675, cy+8, 'â', 30, anchor='middle', italic=True)
    loss(x+673, y+148, '{L_act}(â, a)', 'orange')
    text(x+24, y+156, 'Target: GT navigation action a', 22, C['muted'])

    # Pixel.
    x = 808
    card(x, y, 768, 183, 'Pixel', 'blue', 'Future-frame reconstruction')
    end = pair(x+24, cy)
    arrow(end+12, cy, x+192, cy)
    block(x+198, cy-28, 118, 56, '{E_LAM}')
    arrow(x+322, cy, x+363, cy)
    text(x+384, cy+8, 'z', 29, C['purple'], anchor='middle', italic=True)
    arrow(x+406, cy, x+450, cy)
    block(x+456, cy-28, 123, 56, '{D_pix}', 'blue')
    arrow(x+585, cy, x+629, cy)
    scene(x+642, cy-21, 58, 40, True)
    text(x+671, cy+49, '{x̂_t+Δt}', 24, anchor='middle')
    path([(x+44,cy+44),(x+44,y+151),(x+517,y+151),(x+517,cy+34)], arrow=True)
    text(x+291, y+145, 'Start-frame condition {x_t}', 22, C['muted'], anchor='middle')
    loss(x+675, y+170, '{L_pix}', 'blue')

    # DINO.
    x, y = 24, 268
    card(x, y, 768, 207, 'DINO', 'teal', 'Future-feature reconstruction')
    cy = y+95
    end = pair(x+19, cy, True)
    arrow(end+8,cy,x+119,cy)
    block(x+125,cy-28,87,56,'DINO','teal',size=23)
    arrow(x+218,cy,x+244,cy)
    feature(x+253,cy-29,27)
    feature(x+253,cy+7,27)
    text(x+292,cy-7,'{f_t}',21)
    text(x+292,cy+30,'{f_t+Δt}',21)
    arrow(x+340,cy,x+365,cy)
    block(x+371,cy-28,106,56,'{E_LAM}',size=24)
    arrow(x+483,cy,x+510,cy)
    text(x+530,cy+8,'z',29,C['purple'],anchor='middle',italic=True)
    arrow(x+550,cy,x+576,cy)
    block(x+582,cy-28,76,56,'{D_f}','teal')
    arrow(x+664,cy,x+687,cy)
    feature(x+701,cy-19,37)
    text(x+721,cy+46,'{f̂_t+Δt}',23,anchor='middle')
    path([(x+266,cy-30),(x+266,cy-43),(x+345,cy-43),(x+345,y+174),
          (x+620,y+174),(x+620,cy+34)],arrow=True)
    text(x+478,y+168,'Start feature {f_t}',22,C['muted'],anchor='middle')
    loss(x+718,y+188,'{L_feat}','teal')
    text(x+22,y+188,'{f_t} = DINO({x_t})',22,C['muted'])

    # Pixel + action.
    x = 808
    card(x,y,768,207,'Pixel + action','purple','Joint visual and action supervision')
    cy = y+113
    end=pair(x+24,cy)
    arrow(end+12,cy,x+192,cy)
    block(x+198,cy-28,118,56,'{E_LAM}')
    arrow(x+322,cy,x+363,cy)
    text(x+384,cy+8,'z',29,C['purple'],anchor='middle',italic=True)
    path([(x+406,cy),(x+432,cy),(x+432,y+77),(x+451,y+77)],arrow=True)
    path([(x+432,cy),(x+432,y+143),(x+451,y+143)],arrow=True)
    block(x+457,y+53,123,48,'{D_pix}','blue')
    block(x+457,y+119,123,48,'{D_a}','orange')
    arrow(x+586,y+77,x+624,y+77)
    arrow(x+586,y+143,x+624,y+143)
    text(x+671,y+85,'{x̂_t+Δt}',26,anchor='middle')
    text(x+671,y+151,'â',28,anchor='middle',italic=True)
    path([(x+44,cy+44),(x+44,y+177),(x+744,y+177),(x+744,y+42),
          (x+517,y+42),(x+517,y+47)],arrow=True)
    rect(x+210,y+166,220,25,'white',radius=0)
    text(x+320,y+184,'Start frame {x_t}',22,C['muted'],anchor='middle')
    loss(x+609,y+198,'{L_pix} + λ{L_act}','purple')

    # Explicitly separate the extractor from the NWM's Ez.
    rect(24,492,1552,54,C['purple_bg'],radius=10)
    text(43,527,'TRANSFER',22,C['purple'],700)
    text(191,527,'Frame pair',24)
    arrow(311,519,357,519,color=C['purple'])
    text(376,527,'{E_LAM}',26,C['purple'])
    arrow(464,519,508,519,color=C['purple'])
    text(529,527,'z',29,C['purple'],italic=True)
    arrow(556,519,599,519,color=C['purple'])
    text(621,527,'Latent labels for video pretraining',24)
    text(1556,527,'{E_z}: NWM latent-action encoder',23,C['purple'],anchor='end')


def panel_b():
    section(590,'b','NWM architecture and pretraining','Alternative action interfaces · same backbone architecture')
    for x,w,title,sub,kind in [
        (24,619,'Base data · GT navigation actions','RECON · SCAND · TartanDrive · HuRoN','orange'),
        (658,420,'NavAnywhere v1','13 datasets · 322 h','blue'),
        (1093,483,'NavAnywhere v2 · scale extension','v1 + Ego4D + GO · 1200+ h','teal'),
    ]:
        rect(x,612,w,73,C[kind+'_bg'],radius=9)
        text(x+16,641,title,24,C[kind],700)
        text(x+16,669,sub,23,C['muted'])

    # Four mutually exclusive motion branches, not a mixture of action sources.
    recipes=[('nwm-real','GT actions · no pretraining','Train on Base from scratch','orange'),
             ('nwm-timept','Pretrain on NavAnywhere v1','Base: Reset fine-tuning','blue'),
             ('nwm-geopt','Pretrain on NavAnywhere v1','Base: Reset fine-tuning','teal'),
             ('nwm-latentpt','Pretrain on NavAnywhere v1','Base: strategies in (c)','purple')]
    for i,(name,sub,ft,kind) in enumerate(recipes):
        x,y=24+i*394,707
        rect(x,y,370,167,'white',C['border'],10)
        rect(x,y,370,5,C[kind],radius=2)
        text(x+17,y+33,name,25,C[kind],700)
        text(x+17,y+60,sub,22,C['muted'])
        text(x+185,y+151,ft,22,C['muted'],anchor='middle')
        cy=813
        if i==1:
            text(x+185,cy+7,'No action branch',25,C['blue'],anchor='middle')
            continue
        if i==0:
            text(x+82,cy+8,'a',29,C['orange'],anchor='middle',italic=True)
            arrow(x+106,cy,x+173,cy)
        else:
            block(x+15,cy-25,91,50,'VGGT' if i==2 else '{E_LAM}',kind,size=23)
            arrow(x+112,cy,x+134,cy,dash='5 4')
            text(x+150,cy+8,'g' if i==2 else 'z',27,C[kind],anchor='middle',italic=True)
            arrow(x+166,cy,x+176,cy)
        block(x+182,cy-25,87,50,['{E_a}','','{E_g}','{E_z}'][i],kind,size=26)
        arrow(x+275,cy,x+298,cy)
        text(x+327,cy+8,'{h_m}',26,C[kind],anchor='middle')
        path([(x+327,cy+15),(x+327,892)],color=C['purple'])
    path([(351,892),(1533,892)],color=C['purple'])
    text(24,916,'Dashed arrows: offline video labeling',21,C['muted'])
    text(1090,916,'Select one motion branch; TimePT omits {h_m}',22,C['purple'],anchor='middle')

    # Diffusion step and prediction horizon have distinct learned embeddings.
    text(24,952,'Diffusion step τ',22)
    arrow(178,944,197,944)
    block(203,920,83,48,'{E_τ}','blue')
    text(321,952,'Frame offset Δt',22)
    arrow(469,944,486,944)
    block(492,920,91,48,'{E_Δ}','blue')
    path([(292,944),(306,944),(306,980),(627,980),(627,966)],arrow=True)
    arrow(589,944,607,944)
    arrow(627,892,627,924,color=C['purple'])
    add(f'<circle cx="627" cy="944" r="16" fill="white" stroke="{C["purple"]}" stroke-width="2"/>')
    text(627,953,'+',30,C['purple'],anchor='middle')
    arrow(649,944,711,944,color=C['purple'])
    text(680,935,'c',25,C['purple'],anchor='middle',italic=True)
    rect(718,920,858,49,C['purple_bg'],C['purple'],8)
    text(1147,952,'adaLN conditioning: shift / scale · residual gates in CDiT blocks',24,C['purple'],anchor='middle')

    text(24,1018,'NWM {F_θ}: visual latent diffusion',25,C['blue'],700)
    # Target tokens traverse SA -> CA -> FFN. Context tokens are K/V only.
    rect(719,1006,598,133,C['blue_bg'],C['blue'],10)
    text(912,1035,'CDiT block × L',23,C['blue'],700,anchor='middle')
    for cx in [816,1012,1211]:
        arrow(cx,975,cx,1046,color=C['purple'],width=1.7)
    arrow(1455,975,1455,1034,color=C['purple'],width=1.7)

    # Training target -> visual latent u0 -> forward diffusion -> noisy tokens.
    scene(46,1056,49,37,True)
    text(70,1047,'Target',21,anchor='middle')
    text(70,1122,'train only',21,C['muted'],anchor='middle')
    arrow(101,1076,119,1076)
    block(125,1048,125,56,'VAE','blue',frozen=True,size=24,subtitle='encoder')
    arrow(256,1076,272,1076)
    text(291,1083,'{u_0}',25,anchor='middle')
    arrow(308,1076,325,1076)
    block(331,1050,112,52,'Add noise','blue',size=22)
    text(387,1040,'τ, ε',23,C['muted'],anchor='middle')
    arrow(387,1044,387,1048)
    arrow(449,1076,466,1076)
    text(484,1083,'{u_τ}',25,anchor='middle')
    arrow(503,1076,519,1076)
    block(525,1048,145,56,'Patch embed','blue',size=22,subtitle='+ position')
    arrow(676,1076,735,1076)
    block(741,1052,149,49,'Self-attn','blue',size=23)
    arrow(896,1076,922,1076)
    text(909,1065,'Q',20,C['muted'],anchor='middle')
    block(928,1052,168,49,'Cross-attn','blue',size=23)
    arrow(1102,1076,1137,1076)
    block(1143,1052,136,49,'FFN','blue',size=24)
    arrow(1285,1076,1350,1076)
    block(1356,1040,198,73,'Final projection','blue',size=24,subtitle='+ unpatchify')
    text(737,1128,'Residual + gating',21,C['blue'])

    # Clean context uses the same VAE and patch embedding weights.
    text(67,1165,'Context',21,anchor='middle')
    scene(30,1177,33,28)
    scene(70,1177,33,28,True)
    arrow(109,1190,119,1190)
    block(125,1162,125,56,'VAE','blue',frozen=True,size=24,subtitle='encoder')
    arrow(256,1190,321,1190)
    text(363,1198,'{u_ctx}',25,anchor='middle')
    arrow(405,1190,519,1190)
    block(525,1162,145,56,'Patch embed','blue',size=22,subtitle='+ position')
    path([(676,1190),(1012,1190),(1012,1107)],arrow=True)
    text(814,1180,'Clean context tokens',22,C['muted'],anchor='middle')
    text(1030,1160,'K, V',22,C['blue'])

    arrow(1455,1119,1455,1148)
    text(1455,1178,'{ε̂_θ}, {Σ̂_θ}',28,anchor='middle')
    arrow(1455,1189,1455,1209)
    text(1455,1235,'{L_NWM} · diffusion loss',23,C['blue'],anchor='middle')
    text(24,1243,'Shared, frozen VAE · shared patch embedding · context enters every cross-attention block',22,C['muted'])

    # Generation is separate from the training graph: no future-frame input.
    rect(24,1261,1552,71,C['neutral'],C['border'],9)
    text(43,1291,'GENERATION',22,C['blue'],700)
    text(43,1318,'at inference',21,C['muted'])
    text(302,1304,'Gaussian noise',23,anchor='middle')
    arrow(391,1296,430,1296)
    block(436,1273,317,47,'Iterative denoising with {F_θ}','blue',size=23)
    arrow(759,1296,809,1296)
    text(841,1304,'{û_0}',27,anchor='middle')
    arrow(867,1296,918,1296)
    block(924,1271,206,50,'VAE decoder','blue',frozen=True,size=23)
    arrow(1136,1296,1192,1296)
    scene(1204,1279,50,36,True)
    text(1388,1304,'Future frame {x̂_t+Δt}',25,anchor='middle')
    text(24,1364,'The same context and time / action conditions guide each denoising step.  u: visual latent; z: latent action.',22,C['muted'])


def train_legend(x,y):
    rect(x,y-16,29,21,C['orange_bg'],C['orange'],4,1.8)
    text(x+39,y+2,'Trainable',21,C['muted'])
    rect(x+143,y-16,29,21,C['frozen'],C['muted'],4,1.8,'4 3')
    text(x+182,y+2,'Frozen',21,C['muted'])
    lock(x+261,y-16)


def ft_flow(x,cy,mode,stage):
    if mode=='reset':
        text(x+28,cy+8,'a',27,italic=True)
        arrow(x+49,cy,x+77,cy)
        block(x+84,cy-26,91,52,'{E_a}','orange')
        arrow(x+182,cy,x+251,cy)
        block(x+258,cy-29,126,58,'{F_θ}','blue',frozen=stage==1)
        arrow(x+391,cy,x+428,cy)
        text(x+455,cy+7,'x̂',27,anchor='middle',italic=True)
    elif mode=='a2l':
        text(x+22,cy+8,'a',27,italic=True)
        arrow(x+40,cy,x+57,cy)
        block(x+63,cy-26,71,52,'{E_a}','orange',size=24)
        arrow(x+140,cy,x+157,cy)
        text(x+174,cy+8,'ẑ',27,C['purple'],anchor='middle',italic=True)
        arrow(x+193,cy,x+209,cy)
        block(x+215,cy-26,72,52,'{E_z}','purple',frozen=stage==1,size=24)
        arrow(x+294,cy,x+317,cy)
        block(x+323,cy-29,89,58,'{F_θ}','blue',frozen=stage==1,size=24)
        arrow(x+419,cy,x+444,cy)
        text(x+466,cy+7,'x̂',27,anchor='middle',italic=True)


def align_flow(x,cy,stage):
    # The frozen Ez is a teacher branch, never inserted in the action path.
    if stage==1:
        text(x+27,cy+7,'a',26,italic=True)
        arrow(x+47,cy,x+71,cy)
        block(x+77,cy-22,93,44,'{E_a}','orange',size=24)
        arrow(x+177,cy,x+231,cy)
        text(x+259,cy+7,'{h_a}',25,anchor='middle')
        text(x+27,cy+66,'z',26,C['purple'],italic=True)
        arrow(x+47,cy+59,x+71,cy+59)
        block(x+77,cy+37,93,44,'{E_z}','purple',frozen=True,size=24)
        arrow(x+177,cy+59,x+231,cy+59)
        text(x+259,cy+66,'{h_z}',25,anchor='middle')
        path([(x+286,cy),(x+309,cy),(x+309,cy+59),(x+286,cy+59)],dash='5 4',color=C['purple'])
        text(x+331,cy+32,'{L_align}',24,C['purple'])
        text(x+333,cy+73,'NWM frozen',21,C['muted'])
    else:
        text(x+27,cy+7,'a',26,italic=True)
        arrow(x+47,cy,x+71,cy)
        block(x+77,cy-22,93,44,'{E_a}','orange',size=24)
        arrow(x+177,cy,x+292,cy)
        block(x+299,cy-26,111,52,'{F_θ}','blue',size=24)
        arrow(x+417,cy,x+443,cy)
        text(x+469,cy+7,'x̂',26,anchor='middle',italic=True)
        text(x+27,cy+66,'z',26,C['purple'],italic=True)
        arrow(x+47,cy+59,x+71,cy+59)
        block(x+77,cy+37,93,44,'{E_z}','purple',frozen=True,size=24)
        arrow(x+177,cy+59,x+215,cy+59)
        text(x+244,cy+67,'{h_z}',25,anchor='middle')
        path([(x+244,cy+40),(x+244,cy+7)],dash='5 4',color=C['purple'])
        text(x+261,cy+36,'{L_align}',22,C['purple'])


def panel_c():
    section(1030,'c','Two-stage fine-tuning',None)
    text(24,1064,'From nwm-latentpt · supervised on Base data',24,C['muted'])
    train_legend(1288,1059)
    cols=[(24,'Reset','Replace {E_z} with {E_a}','reset','orange'),
          (552,'Align','Keep {E_z} as a frozen teacher','align','purple'),
          (1080,'Action2Latent','Map actions into the latent input space','a2l','teal')]
    for x,title,desc,mode,kind in cols:
        rect(x,1087,496,419,'white',C['border'],11)
        rect(x,1087,496,5,C[kind],radius=2)
        text(x+20,1124,title,27,C[kind],700)
        if mode=='reset':
            text(x+475,1123,'baseline',22,C['muted'],anchor='end')
        text(x+20,1154,desc,22,C['muted'])
        text(x+20,1191,'Step 1',23,weight=700)
        text(x+475,1191,'Train {E_a} only',23,C['muted'],anchor='end')
        if mode=='align':
            align_flow(x,1230,1)
        else:
            ft_flow(x,1243,mode,1)
            loss(x+248,1304,'{L_NWM}',size=24)
        path([(x+19,1324),(x+477,1324)],color=C['border'],width=1.2)
        text(x+20,1357,'Step 2',23,weight=700)
        train='Train all except {E_z}' if mode=='align' else 'Train all modules'
        text(x+476,1357,train,22,C['muted'],anchor='end')
        if mode=='align':
            align_flow(x,1394,2)
            loss(x+248,1491,'{L_NWM} + λ{L_align}',size=23)
        else:
            ft_flow(x,1410,mode,2)
            loss(x+248,1478,'{L_NWM}',size=24)
    text(24,1538,'Training-only teacher in Align; Action2Latent retains {E_z} at inference.  Visual / diffusion inputs are omitted in (c).',23,C['muted'])
    path([(24,1553),(1576,1553)],color=C['border'],width=1.2)
    text(800,1579,'EVALUATION   Visual prediction  ·  Navigation planning  ·  Unseen-domain generalization',23,C['ink'],anchor='middle')


DEFS = '''<defs>
  <marker id="default" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9 Z" fill="#64748B"/></marker>
  <marker id="purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9 Z" fill="#7755A6"/></marker>
</defs>'''


def export(name, content, top, height):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="6.75in" '
           f'height="{6.75*height/W:.6f}in" viewBox="0 {top} {W} {height}">'
           f'<title>LAM and NWM training paradigms — {name}</title>'
           '<desc>Four latent action model objectives, four NWM pretraining '
           'recipes and three two-stage fine-tuning strategies.</desc>'
           + DEFS + f'<rect x="0" y="{top}" width="1600" height="{height}" fill="white"/>'
           + content + '</svg>')
    (OUT/f'{name}.svg').write_text(svg, encoding='utf-8')
    cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(OUT/f'{name}.pdf'))
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/f'{name}.png'),output_width=3200)
    print(f'Wrote {name}.svg / .pdf / .png')


def main():
    sections=[]
    for fn in [panel_a,panel_b,panel_c]:
        start=len(parts)
        fn()
        sections.append(''.join(parts[start:]))
    # Expand the NWM panel while retaining the original standalone FT layout.
    shifted_c='<g transform="translate(0 390)">'+sections[2]+'</g>'
    export('overview',sections[0]+sections[1]+shifted_c,0,H)
    export('panel_a_lam',sections[0],0,560)
    export('panel_b_pretraining',sections[1],553,826)
    export('panel_c_finetuning',sections[2],995,599)


if __name__=='__main__':
    main()
