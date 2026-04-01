# R-MoE Docker Image
FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="R-MoE"
LABEL org.opencontainers.image.description="Recursive Multi-Agent Mixture-of-Experts for Clinical Diagnostics"
LABEL org.opencontainers.image.vendor="R-MoE Team"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 rmoe
USER rmoe
WORKDIR /home/rmoe

# Copy binary
COPY --chmod=755 rmoe-linux-x86_64 /usr/local/bin/rmoe

# Create config directory
RUN mkdir -p /home/rmoe/.rmoe

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD rmoe --version || exit 1

# Default command
ENTRYPOINT ["rmoe"]
CMD ["--help"]

# Expose API port
EXPOSE 8080
