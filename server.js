  res.json({
    totalChecks: classifications.length,
    feedbackReceived: answered,
    markedCorrect: correctCount,
    markedWrong: answered - correctCount,
    percentCorrectOfAnswered: answered ? Math.round((correctCount / answered) * 100) : null,
    checksByDiagnosis: byLabel,
  });
});

app.get("/export.csv", requireAdmin, (req, res) => {
  const classifications = readLog(CLASSIFY_LOG);
  const latestFeedback = new Map();
  readLog(FEEDBACK_LOG).forEach((f) => latestFeedback.set(f.id, f.correct));

  const rows = ["id,time,diagnosis,confidence,farmer_said_correct"];
  classifications.forEach((c) => {
    const fb = latestFeedback.has(c.id) ? (latestFeedback.get(c.id) ? "yes" : "no") : "";
    const label = String(c.label).replace(/"/g, '""');
    rows.push(`${c.id},${c.time},"${label}",${(c.confidence * 100).toFixed(1)},${fb}`);
  });

  res.setHeader("Content-Type", "text/csv");
  res.setHeader("Content-Disposition", "attachment; filename=easifarm-pilot.csv");
  res.send(rows.join("\n"));
});

// ---------- Start ----------
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
