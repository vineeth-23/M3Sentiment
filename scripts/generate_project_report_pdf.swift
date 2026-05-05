import AppKit
import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let outURL = root.appendingPathComponent("reports/M3Sentiment_Detailed_Project_Report.pdf")
let plots = root.appendingPathComponent("outputs/plots/diagnostics")

let pageRect = CGRect(x: 0, y: 0, width: 792, height: 612)
let margin: CGFloat = 44

let titleFont = NSFont.systemFont(ofSize: 28, weight: .bold)
let subtitleFont = NSFont.systemFont(ofSize: 14, weight: .medium)
let headingFont = NSFont.systemFont(ofSize: 19, weight: .bold)
let bodyFont = NSFont.systemFont(ofSize: 10.8, weight: .regular)
let boldFont = NSFont.systemFont(ofSize: 10.8, weight: .semibold)
let smallFont = NSFont.systemFont(ofSize: 8.8, weight: .regular)
let monoFont = NSFont.monospacedSystemFont(ofSize: 9.3, weight: .regular)

let ink = NSColor(calibratedRed: 0.07, green: 0.10, blue: 0.16, alpha: 1.0)
let muted = NSColor(calibratedRed: 0.35, green: 0.41, blue: 0.50, alpha: 1.0)
let blue = NSColor(calibratedRed: 0.10, green: 0.32, blue: 0.85, alpha: 1.0)
let teal = NSColor(calibratedRed: 0.02, green: 0.48, blue: 0.45, alpha: 1.0)
let softBlue = NSColor(calibratedRed: 0.91, green: 0.96, blue: 1.0, alpha: 1.0)
let softGreen = NSColor(calibratedRed: 0.92, green: 0.98, blue: 0.94, alpha: 1.0)
let softOrange = NSColor(calibratedRed: 1.0, green: 0.96, blue: 0.88, alpha: 1.0)

func attrs(_ font: NSFont, _ color: NSColor = ink, align: NSTextAlignment = .left) -> [NSAttributedString.Key: Any] {
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = align
    paragraph.lineSpacing = 2.4
    return [.font: font, .foregroundColor: color, .paragraphStyle: paragraph]
}

func drawText(_ text: String, in rect: CGRect, font: NSFont = bodyFont, color: NSColor = ink, align: NSTextAlignment = .left) {
    text.draw(in: rect, withAttributes: attrs(font, color, align: align))
}

@discardableResult
func drawWrapped(_ text: String, x: CGFloat, y: CGFloat, width: CGFloat, font: NSFont = bodyFont, color: NSColor = ink) -> CGFloat {
    let attributed = NSAttributedString(string: text, attributes: attrs(font, color))
    let height = attributed.boundingRect(
        with: CGSize(width: width, height: 1000),
        options: [.usesLineFragmentOrigin, .usesFontLeading]
    ).height + 5
    attributed.draw(in: CGRect(x: x, y: y, width: width, height: height))
    return y + height
}

@discardableResult
func drawBullets(_ bullets: [String], x: CGFloat, y: CGFloat, width: CGFloat, font: NSFont = bodyFont) -> CGFloat {
    var cursor = y
    for bullet in bullets {
        drawText("•", in: CGRect(x: x, y: cursor + 1, width: 12, height: 18), font: boldFont, color: teal)
        cursor = drawWrapped(bullet, x: x + 18, y: cursor, width: width - 18, font: font)
        cursor += 2
    }
    return cursor
}

func drawPill(_ text: String, x: CGFloat, y: CGFloat, width: CGFloat, color: NSColor) {
    color.setFill()
    NSBezierPath(roundedRect: CGRect(x: x, y: y, width: width, height: 28), xRadius: 14, yRadius: 14).fill()
    drawText(text, in: CGRect(x: x + 12, y: y + 6, width: width - 24, height: 18), font: boldFont, color: ink, align: .center)
}

func drawCard(_ rect: CGRect, fill: NSColor = .white) {
    fill.setFill()
    let path = NSBezierPath(roundedRect: rect, xRadius: 14, yRadius: 14)
    path.fill()
    NSColor(calibratedWhite: 0.84, alpha: 1).setStroke()
    path.lineWidth = 0.8
    path.stroke()
}

func drawImage(_ name: String, in rect: CGRect, caption: String? = nil) {
    let url = plots.appendingPathComponent(name)
    guard let image = NSImage(contentsOf: url) else {
        drawCard(rect, fill: NSColor(calibratedWhite: 0.96, alpha: 1.0))
        drawText("Missing image: \(name)", in: rect.insetBy(dx: 12, dy: 12), font: bodyFont, color: muted, align: .center)
        return
    }
    drawCard(rect, fill: .white)
    image.draw(in: rect.insetBy(dx: 8, dy: 8), from: .zero, operation: .sourceOver, fraction: 1.0, respectFlipped: true, hints: nil)
    if let caption {
        drawText(caption, in: CGRect(x: rect.minX, y: rect.maxY + 4, width: rect.width, height: 16), font: smallFont, color: muted, align: .center)
    }
}

func drawTable(_ headers: [String], _ rows: [[String]], x: CGFloat, y: CGFloat, widths: [CGFloat], rowHeight: CGFloat = 24) -> CGFloat {
    let totalWidth = widths.reduce(0, +)
    softBlue.setFill()
    NSBezierPath(roundedRect: CGRect(x: x, y: y, width: totalWidth, height: rowHeight), xRadius: 8, yRadius: 8).fill()
    var cx = x
    for (i, h) in headers.enumerated() {
        drawText(h, in: CGRect(x: cx + 6, y: y + 6, width: widths[i] - 12, height: rowHeight - 8), font: smallFont, color: ink)
        cx += widths[i]
    }
    var cy = y + rowHeight
    for (ridx, row) in rows.enumerated() {
        (ridx % 2 == 0 ? NSColor.white : NSColor(calibratedWhite: 0.975, alpha: 1.0)).setFill()
        NSBezierPath(rect: CGRect(x: x, y: cy, width: totalWidth, height: rowHeight)).fill()
        NSColor(calibratedWhite: 0.88, alpha: 1).setStroke()
        NSBezierPath(rect: CGRect(x: x, y: cy, width: totalWidth, height: rowHeight)).stroke()
        cx = x
        for (i, cell) in row.enumerated() {
            let font = i == 0 ? smallFont : monoFont
            drawText(cell, in: CGRect(x: cx + 6, y: cy + 6, width: widths[i] - 12, height: rowHeight - 8), font: font, color: ink)
            cx += widths[i]
        }
        cy += rowHeight
    }
    return cy
}

func newPage(_ ctx: CGContext, _ pageNo: Int, title: String) {
    ctx.beginPDFPage(nil)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
    NSColor.white.setFill()
    NSBezierPath(rect: pageRect).fill()
    drawText(title, in: CGRect(x: margin, y: 28, width: pageRect.width - 2 * margin, height: 34), font: titleFont, color: ink)
    drawText("M3Sentiment Project Report", in: CGRect(x: margin, y: 63, width: 330, height: 18), font: smallFont, color: muted)
    drawText("Page \(pageNo)", in: CGRect(x: pageRect.width - margin - 70, y: 63, width: 70, height: 18), font: smallFont, color: muted, align: .right)
    NSColor(calibratedWhite: 0.88, alpha: 1).setStroke()
    let line = NSBezierPath()
    line.move(to: CGPoint(x: margin, y: 88))
    line.line(to: CGPoint(x: pageRect.width - margin, y: 88))
    line.stroke()
}

func endPage(_ ctx: CGContext) {
    NSGraphicsContext.restoreGraphicsState()
    ctx.endPDFPage()
}

var mediaBox = pageRect
guard let consumer = CGDataConsumer(url: outURL as CFURL),
      let ctx = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
    fatalError("Could not create PDF context")
}

// Page 1
newPage(ctx, 1, title: "Overview")
drawWrapped("M3Sentiment is a transformer-based multimodal sentiment analysis project on CMU-MOSEI. It predicts whether a video clip is negative, neutral, or positive by combining text, audio, and visual signals.", x: margin, y: 112, width: 420, font: subtitleFont, color: ink)
drawBullets([
    "Main task: 3-way sentiment classification from aligned text, audio, and vision sequences.",
    "Core comparison: Late Fusion, Cross-Modal Attention, Cross-Modal + Orthogonality Loss, and Cross-Modal + Auxiliary Heads.",
    "Main contribution: beyond accuracy, the project tracks internal diagnostics such as modality attention, orthogonality, attention entropy, and loss components."
], x: margin, y: 190, width: 420)
drawImage("test_accuracy_by_model.svg", in: CGRect(x: 500, y: 118, width: 235, height: 175), caption: "Final test accuracy by model")
drawImage("train_accuracy_by_model.svg", in: CGRect(x: 500, y: 330, width: 235, height: 165), caption: "Training accuracy over epochs")
drawPill("Text", x: margin, y: 440, width: 100, color: softBlue)
drawPill("Audio", x: margin + 120, y: 440, width: 100, color: softGreen)
drawPill("Vision", x: margin + 240, y: 440, width: 100, color: softOrange)
drawWrapped("Final takeaway: smarter fusion helps modestly, while diagnostics explain why models behave differently.", x: margin, y: 492, width: 430, font: boldFont, color: teal)
endPage(ctx)

// Page 2
newPage(ctx, 2, title: "Data And Preprocessing")
drawWrapped("The project expects a processed CMU-MOSEI dataset at data/aligned_mosei_dataset.pkl. Each video clip is represented as three aligned sequences with length 50: text, audio, and vision.", x: margin, y: 112, width: 690, font: bodyFont)
drawTable(
    ["Component", "Implementation"],
    [
        ["Dataset loader", "Loads train/valid/test splits from aligned_mosei_dataset.pkl"],
        ["Normalization", "Computes mean/std on train split and reuses them for validation/test"],
        ["Audio cleanup", "Replaces NaN and ±inf audio values with zero"],
        ["Labels", "Uses 3-class labels: negative, neutral, positive"],
        ["Dataloaders", "Train loader shuffles; validation/test loaders are deterministic"]
    ],
    x: margin, y: 172, widths: [145, 540], rowHeight: 30
)
drawWrapped("Implementation files: src/m3sentiment/dataset.py and src/m3sentiment/data_loaders.py. Data utilities for download, CMU Multimodal SDK processing, exporting, and inspection are under data/scripts/.", x: margin, y: 362, width: 690, font: bodyFont)
drawCard(CGRect(x: margin, y: 430, width: 690, height: 84), fill: NSColor(calibratedWhite: 0.975, alpha: 1))
drawText("Data flow", in: CGRect(x: margin + 20, y: 447, width: 90, height: 18), font: boldFont, color: teal)
drawText("CMU-MOSEI video clip  →  aligned text/audio/vision features  →  normalized tensors  →  transformer model", in: CGRect(x: margin + 20, y: 477, width: 650, height: 22), font: monoFont, color: ink)
endPage(ctx)

// Page 3
newPage(ctx, 3, title: "Model Architectures")
drawWrapped("All models share a common transformer-based multimodal backbone. Each modality is projected into a shared hidden dimension, receives positional encoding, passes through transformer encoder layers, and is pooled into one summary vector.", x: margin, y: 112, width: 690)
drawTable(
    ["Model", "What it implements", "Why it matters"],
    [
        ["Late Fusion", "Separate modality encoders plus final fusion transformer", "Simple multimodal baseline"],
        ["Cross-Modal", "Each modality attends to the other two modalities", "Models disagreement and interaction"],
        ["Ortho Fusion", "Cross-modal model plus orthogonality loss", "Reduces redundant modality features"],
        ["Aux Fusion", "Cross-modal model plus text/audio/vision auxiliary heads", "Forces each branch to learn sentiment cues"]
    ],
    x: margin, y: 185, widths: [120, 290, 275], rowHeight: 34
)
drawCard(CGRect(x: margin, y: 360, width: 690, height: 126), fill: NSColor(calibratedWhite: 0.975, alpha: 1))
drawText("Core pipeline", in: CGRect(x: margin + 18, y: 377, width: 120, height: 18), font: boldFont, color: teal)
drawText("Text sequence   → Text Transformer   → Text vector", in: CGRect(x: margin + 22, y: 407, width: 630, height: 18), font: monoFont)
drawText("Audio sequence  → Audio Transformer  → Audio vector", in: CGRect(x: margin + 22, y: 432, width: 630, height: 18), font: monoFont)
drawText("Vision sequence → Vision Transformer → Vision vector", in: CGRect(x: margin + 22, y: 457, width: 630, height: 18), font: monoFont)
drawWrapped("The architecture uses instrumented transformer layers, so attention maps are available for later diagnostic analysis.", x: margin, y: 515, width: 690, font: bodyFont)
endPage(ctx)

// Page 4
newPage(ctx, 4, title: "Training And Loss Functions")
drawWrapped("Training is implemented in scripts/train_models.py and src/m3sentiment/training.py. All models use CrossEntropyLoss, Adam, gradient clipping, and ReduceLROnPlateau scheduling.", x: margin, y: 112, width: 690)
drawTable(
    ["Model", "Training objective"],
    [
        ["Late Fusion", "classification loss"],
        ["Cross-Modal", "classification loss"],
        ["Ortho Fusion", "classification loss + 100 × orthogonality loss"],
        ["Aux Fusion", "main fused loss + 0.05 × (text aux + audio aux + vision aux)"]
    ],
    x: margin, y: 176, widths: [130, 555], rowHeight: 32
)
drawImage("batch_total_loss_by_model.svg", in: CGRect(x: margin, y: 335, width: 310, height: 185), caption: "Batch total loss comparison")
drawImage("ortho_fusion_loss_components.svg", in: CGRect(x: margin + 360, y: 335, width: 310, height: 185), caption: "Ortho loss components")
drawWrapped("Final config: batch size 64, learning rate 2e-5, 40 epochs, hidden dimension 128, 4 attention heads, 2 transformer layers, dropout 0.1.", x: margin, y: 540, width: 690, font: bodyFont, color: muted)
endPage(ctx)

// Page 5
newPage(ctx, 5, title: "Evaluation Results")
drawTable(
    ["Model", "Final Test Acc", "Best Test Acc", "Macro F1", "Weighted F1"],
    [
        ["Late Fusion", "67.03%", "67.44%", "61.22%", "65.66%"],
        ["Cross-Modal", "67.40%", "67.93%", "61.43%", "65.90%"],
        ["Ortho Fusion", "68.02%", "68.17%", "61.79%", "66.41%"],
        ["Aux Fusion", "67.10%", "67.50%", "61.14%", "65.63%"]
    ],
    x: margin, y: 112, widths: [145, 125, 125, 125, 125], rowHeight: 30
)
drawImage("test_accuracy_by_model.svg", in: CGRect(x: margin, y: 290, width: 310, height: 210), caption: "Final test accuracy")
drawImage("test_loss_by_model.svg", in: CGRect(x: margin + 360, y: 290, width: 310, height: 210), caption: "Final test loss")
drawWrapped("Key result: Cross-Modal + Orthogonality achieves the best final accuracy and best F1 values in the finalized run. This suggests that reducing modality redundancy can improve generalization.", x: margin, y: 525, width: 690, font: boldFont, color: teal)
endPage(ctx)

// Page 6
newPage(ctx, 6, title: "Representation Diagnostics")
drawWrapped("Orthogonality diagnostics measure how similar the learned text, audio, and vision vectors are. Lower similarity means the modalities are learning more distinct information.", x: margin, y: 112, width: 690)
drawTable(
    ["Model", "Avg Modality Similarity"],
    [
        ["Late Fusion", "0.2529"],
        ["Cross-Modal", "0.1251"],
        ["Ortho Fusion", "0.0025"],
        ["Aux Fusion", "0.1759"]
    ],
    x: margin, y: 175, widths: [180, 190], rowHeight: 28
)
drawImage("ortho_fusion_orthogonality_over_training.svg", in: CGRect(x: 430, y: 130, width: 300, height: 205), caption: "Orthogonality over training")
drawWrapped("The Orthogonality model successfully drives modality similarity close to zero. This is the clearest evidence that the added loss changed what the model learned internally.", x: margin, y: 345, width: 690, font: boldFont, color: teal)
drawImage("cross_modal_orthogonality_over_training.svg", in: CGRect(x: margin, y: 410, width: 310, height: 150), caption: "Cross-Modal similarity trend")
drawImage("aux_fusion_orthogonality_over_training.svg", in: CGRect(x: margin + 360, y: 410, width: 310, height: 150), caption: "Aux Fusion similarity trend")
endPage(ctx)

// Page 7
newPage(ctx, 7, title: "Attention Diagnostics")
drawWrapped("The diagnostic framework tracks how much attention is paid to each modality and how attention changes across heads, layers, batches, and epochs.", x: margin, y: 112, width: 690)
drawImage("late_fusion_final_fusion_attention.svg", in: CGRect(x: margin, y: 165, width: 210, height: 150), caption: "Late Fusion")
drawImage("cross_modal_final_cross_attention.svg", in: CGRect(x: margin + 240, y: 165, width: 210, height: 150), caption: "Cross-Modal")
drawImage("ortho_fusion_final_cross_attention.svg", in: CGRect(x: margin + 480, y: 165, width: 210, height: 150), caption: "Ortho Fusion")
drawTable(
    ["Observation", "Interpretation"],
    [
        ["Late Fusion attends most to text", "Text is the dominant sentiment signal"],
        ["Cross-Modal attention is more balanced", "Modalities interact more directly"],
        ["Ortho audio/vision attend strongly to text", "Text becomes the anchor after modality separation"],
        ["Head-level plots expose variation", "Different attention heads can specialize differently"]
    ],
    x: margin, y: 365, widths: [250, 435], rowHeight: 31
)
drawImage("ortho_fusion_final_head_attention_heatmap.svg", in: CGRect(x: margin + 455, y: 472, width: 230, height: 85), caption: "Head-level attention heatmap")
endPage(ctx)

// Page 8
newPage(ctx, 8, title: "Failure Analysis")
drawWrapped("Confusion matrices show that all models are much better at detecting positive and negative sentiment than neutral sentiment.", x: margin, y: 112, width: 690)
drawTable(
    ["Model", "Negative Recall", "Neutral Recall", "Positive Recall"],
    [
        ["Late Fusion", "71.63%", "31.12%", "80.43%"],
        ["Cross-Modal", "72.74%", "30.63%", "80.74%"],
        ["Ortho Fusion", "72.67%", "30.15%", "82.27%"],
        ["Aux Fusion", "72.15%", "30.54%", "80.52%"]
    ],
    x: margin, y: 170, widths: [150, 160, 160, 160], rowHeight: 30
)
drawImage("ortho_fusion_test_confusion_matrix.svg", in: CGRect(x: margin, y: 340, width: 300, height: 190), caption: "Ortho Fusion confusion matrix")
drawImage("cross_modal_test_confusion_matrix.svg", in: CGRect(x: margin + 365, y: 340, width: 300, height: 190), caption: "Cross-Modal confusion matrix")
drawWrapped("Main failure mode: neutral examples are often pushed into positive or negative classes because they contain weak or ambiguous emotional cues.", x: margin, y: 548, width: 690, font: boldFont, color: teal)
endPage(ctx)

// Page 9
newPage(ctx, 9, title: "Conclusions And Future Work")
drawBullets([
    "The project implements four transformer-based multimodal sentiment models and compares them under the same training setup.",
    "Cross-Modal + Orthogonality achieves the best final test accuracy in the finalized outputs.",
    "Orthogonality diagnostics confirm that the orthogonality loss almost completely separates text, audio, and vision representations.",
    "Attention diagnostics show that text remains a strong anchor, especially for audio and vision queries.",
    "Neutral sentiment is the hardest class and is the main opportunity for future improvement."
], x: margin, y: 115, width: 690)
drawText("Future work", in: CGRect(x: margin, y: 320, width: 220, height: 24), font: headingFont, color: ink)
drawBullets([
    "Tune orthogonality and auxiliary loss weights.",
    "Use class-balanced loss or sampling to improve neutral sentiment.",
    "Select best-validation checkpoints instead of only final epoch weights.",
    "Run multiple random seeds for stronger statistical confidence.",
    "Try larger pretrained multimodal models and add qualitative example-level analysis."
], x: margin, y: 360, width: 690)
drawCard(CGRect(x: margin, y: 505, width: 690, height: 48), fill: softGreen)
drawText("Final takeaway: M3Sentiment shows that internal diagnostics are as important as accuracy for understanding multimodal sentiment models.", in: CGRect(x: margin + 18, y: 522, width: 654, height: 20), font: boldFont, color: ink, align: .center)
endPage(ctx)

ctx.closePDF()
print("Saved PDF to \(outURL.path)")
