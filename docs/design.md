# Atlas Deconvolve: Web Application Design

This document outlines the architecture for the Atlas Deconvolve web platform. It details two potential architectural designs:

*   **V1: Static Analysis Portal:** A simple, synchronous application for exploring pre-computed results from a single, static PPI network.
*   **V2: Dynamic Analysis Platform:** A powerful, asynchronous, job-based platform that allows users to upload their own networks and receive results via email.

---

## Architecture V1: Static Analysis Portal

This architecture is for a simple, fast website where the underlying PPI network and GVAE model are static and pre-trained.

### V1: Components

*   **Frontend (React.js):** A single-page application where a user can select a pre-defined pathway (e.g., "FGFR Signaling").
*   **Backend (Python/FastAPI):** A simple API server that, upon request:
    1.  Loads the pre-trained embeddings (`final_embeddings.pth`) and network data.
    2.  Fetches pathway members from g:Profiler.
    3.  Runs the prediction logic.
    4.  Returns the results immediately in a single HTTP response.
*   **Data Layer:** A static set of files: `final_embeddings.pth`, `node2idx.json`, and `edgelist.tsv`.

### V1: Workflow

1.  User selects "FGFR Signaling" and clicks "Analyze".
2.  Frontend sends a request to the backend.
3.  Backend computes the result in a few seconds and sends it back.
4.  Frontend displays the visualization.

---

## Architecture V2: Dynamic Analysis Platform

This is a more advanced, scalable architecture designed to handle user-uploaded networks and on-demand model training, which can be a very time-consuming process.

### V2: Frontend (New User Experience)

1.  **The "Deconvolve" Page becomes a "Submit Job" Form:**
    *   **File Upload:** A field to upload a PPI network file (e.g., `.tsv` or `.csv`).
    *   **Target Input:** A text field for the user to input the target for analysis (a disease, pathway, or single protein name).
    *   **Email Field (New):** An input labeled `Email for Notification (Optional)`.
    *   The `[ Run Analysis ]` button becomes `[ Submit Analysis Job ]`.

2.  **A "Dashboard" or "My Analyses" Page:**
    *   This page lists all submitted jobs and their real-time status: `Queued`, `Preprocessing`, `Training Model`, `Predicting Interactions`, `Complete`, or `Failed`.
    *   Completed jobs have a `[ View Results ]` button that links to a unique results page.

### V2: Backend (Asynchronous & Job-Based)

The backend is split into three distinct parts to manage long-running tasks without blocking the user interface.

1.  **Web Server (The "Receptionist"):**
    *   **Technology:** FastAPI.
    *   **Role:**
        *   Handles user authentication and file uploads.
        *   Validates the user's input.
        *   Creates a "job" record in a database (including the optional email).
        *   Places this job onto the **Job Queue**.
        *   Provides API endpoints for the frontend to fetch job statuses for the Dashboard.

2.  **Job Queue (The "To-Do List"):**
    *   **Technology:** A dedicated message queue system like **RabbitMQ** or **Redis**.
    *   **Role:** Decouples the fast web server from the slow analysis tasks. The web server can instantly respond to the user ("Job Submitted!") after adding the task to this queue.

3.  **Worker(s) (The "Heavy Lifters"):**
    *   **Technology:** Separate Python processes (running on dedicated machines, potentially with GPUs).
    *   **Role:** These continuously monitor the Job Queue for new tasks. When a job appears, a worker executes the full pipeline:
        1.  **Preprocess:** Convert the user's uploaded file into the required graph format.
        2.  **Train:** Train the GVAE model **from scratch** on the new data, generating a new, custom set of embeddings.
        3.  **Predict:** Run the novel interaction prediction logic against the user's specified target.
        4.  **Store Results:** Save the final results (JSON for the graph/table) to a database or file storage.
        5.  **Update Status:** Mark the job as `Complete` in the database.
        6.  **Send Notification:** If an email was provided, connect to an Email Sending Service and send the user an email with a direct link to their results page.

### V2: New Required Technologies

*   **Job Queue:** RabbitMQ, Redis, or Celery.
*   **Database:** PostgreSQL or similar, to track users and analysis jobs.
*   **File Storage:** A local filesystem or a cloud solution like Amazon S3 to store user-uploaded files and results.
*   **Email Sending Service:** A transactional email API like **SendGrid**, **Mailgun**, or **Amazon SES** to reliably send notifications.

### V2: End-to-End Workflow

1.  User uploads their PPI network, enters "melanoma" as the target and their email, and clicks "Submit Analysis Job".
2.  The FastAPI server validates the input, creates a job record, and pushes the job to the **RabbitMQ queue**. It immediately responds to the frontend that the job was submitted successfully.
3.  The user is redirected to their Dashboard, where they can see their new job with the status `Queued`. They can now close their browser.
4.  A **Worker** process picks up the job from the queue and begins the hours-long process of training and analysis. It periodically updates the job's status in the database.
5.  Once the analysis is complete, the worker saves the results and sends an email to the user via **SendGrid**.
6.  The user receives the email, clicks the link, and is taken directly to the results page, which displays their custom-generated network visualization.
