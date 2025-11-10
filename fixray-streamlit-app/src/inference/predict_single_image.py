import os
def load_model(model_path, device):
    """
    Lädt ein trainiertes Modell und unterstützt mehrere Checkpoint-Formate:
      - vollständiges checkpoint dict mit 'model_state_dict' oder 'state_dict'
      - reines state_dict (mapping param_name -> tensor)
      - (selten) ein gespeichertes nn.Module-Objekt
    Liefert immer ein torch.nn.Module im eval()-Modus zurück.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

    logger.info(f"Lade Modell von: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Falls checkpoint direkt ein nn.Module ist
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        logger.info("Checkpoint ist ein nn.Module -> direkt verwendet")
        return model

    # Sonst bauen wir das Modell auf und versuchen, state_dict zu finden
    model = get_model(CONFIG['num_classes'])

    state_dict = None
    if isinstance(checkpoint, dict):
        # gängige Schlüssel prüfen
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Heuristik: wenn Werte Tensoren sind, behandeln wir dict als state_dict
            try:
                first_val = next(iter(checkpoint.values()))
                if isinstance(first_val, torch.Tensor):
                    state_dict = checkpoint
            except StopIteration:
                state_dict = None

    if state_dict is None:
        raise RuntimeError(
            "Unbekanntes Checkpoint-Format. Erwartet 'model_state_dict'/'state_dict' oder ein state_dict. "
            "Führe: ck = torch.load(...); print(type(ck)); print(list(ck.keys())[:20])"
        )

    # Entferne ggf. 'module.' Präfixe aus geladenem state_dict (z.B. bei DataParallel)
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith('module.'):
        new_state = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        state_dict = new_state

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Modell erfolgreich geladen und in den eval()-Modus gesetzt")
    return model

def analyze_image(image_path, model, output_dir, confidence_threshold=0.5):
    """
    Führt Inferenz auf einem einzelnen Bild durch, speichert ein annotiertes Ergebnisbild
    und gibt eine Ergebnis-Zusammenfassung zurück.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    logger.info(f"Analysiere Bild: {image_path}")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Vorbereitung wie beim Training: Tensor + Normalisierung
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(CONFIG['device'])

    with torch.no_grad():
        outputs = model(input_tensor)

    if not outputs:
        return {'status': 'ERROR', 'error_message': 'Leere Ausgabe vom Modell', 'image_path': image_path}

    out = outputs[0]
    boxes = out.get('boxes', torch.empty((0,4))).cpu().numpy()
    scores = out.get('scores', torch.empty((0,))).cpu().numpy()
    labels = out.get('labels', torch.empty((0,))).cpu().numpy()
    masks = out.get('masks', None)
    if masks is not None:
        masks = masks.cpu().numpy()

    # Filter nach Schwelle
    keep = scores >= confidence_threshold
    boxes_f = boxes[keep]
    scores_f = scores[keep]
    labels_f = labels[keep]
    masks_f = masks[keep] if masks is not None else None

    fractures_detected = int(len(boxes_f))

    # Annotation auf Bild zeichnen
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes_f):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        label_txt = f"Fraktur {scores_f[i]:.2f}"
        draw.rectangle([x1, y1-16, x1+len(label_txt)*7, y1], fill='red')
        draw.text((x1+2, y1-14), label_txt, fill='white', font=font)

        # falls Maske vorhanden, einfache Paste (resized)
        if masks_f is not None:
            mask = masks_f[i, 0]
            mask_img = Image.fromarray((mask * 255).astype('uint8')).resize((w, h)).convert("L")
            red = Image.new("RGBA", (w, h), (255,0,0,80))
            draw_img = Image.composite(red, draw_img.convert("RGBA"), mask_img).convert("RGB")

    # Ausgabe speichern
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"fracture_analysis_{timestamp}.png")
    draw_img.save(out_path)

    summary = {
        "status": "SUCCESS",
        "image_path": image_path,
        "fractures_detected": fractures_detected,
        "confidence_scores": scores_f.tolist(),
        "output_image": out_path
    }
    logger.info(f"Analyse fertig: {fractures_detected} Treffer. Ergebnis: {out_path}")
    return summary