# Runs the offline demo (no API key, no cost). Works on a free Hugging Face Space.
FROM python:3.12-slim

RUN useradd -m -u 1000 app
WORKDIR /home/app/shadow-eval
COPY --chown=app . .
RUN pip install --no-cache-dir -e .

USER app
ENV PORT=7860
EXPOSE 7860

# Build the demo database on start-up, then serve the review UI, dashboard and report.
CMD shadow demo --db /tmp/demo.db --report /tmp/demo_report.html && \
    shadow serve --db /tmp/demo.db --host 0.0.0.0 --port ${PORT} \
      --banner "Live demo · synthetic insurance claims · simulated agent and expert · resets on restart"
