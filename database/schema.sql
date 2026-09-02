-- database/schema.sql

-- Minimal professional schema for MGC lead tracking.
-- Keeping the schema simple and directly tied to the analytical requirements.

CREATE TABLE IF NOT EXISTS leads (
    lead_id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    
    -- Lead Profile & Requirements
    source VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    area VARCHAR(100),
    property_type VARCHAR(50) NOT NULL,
    budget_pkr_lac DECIMAL(10, 2),
    bedrooms INT,
    is_overseas INT NOT NULL DEFAULT 0,
    referred_by_existing_client INT NOT NULL DEFAULT 0,
    has_financing_approved INT NOT NULL DEFAULT 0,
    
    -- Engagement & Post-Contact metrics
    first_response_minutes DECIMAL(10, 2),
    calls_made INT NOT NULL DEFAULT 0,
    total_call_seconds DECIMAL(10, 2) NOT NULL DEFAULT 0,
    whatsapp_replies INT NOT NULL DEFAULT 0,
    site_visits INT NOT NULL DEFAULT 0,
    agent_experience_years DECIMAL(4, 2),
    token_amount_received_pkr DECIMAL(15, 2) NOT NULL DEFAULT 0,
    
    -- Outcome & Identity
    -- Ideally, to prevent duplicates at the database level, we could add a UNIQUE constraint 
    -- on a natural key (like phone number/email) or on the crm_record_hash.
    -- E.g.: CONSTRAINT uq_crm_hash UNIQUE (crm_record_hash)
    -- However, since the dataset contains duplicates for the task, we index it instead.
    crm_record_hash VARCHAR(64) NOT NULL,
    converted INT NOT NULL DEFAULT 0
);

CREATE INDEX idx_leads_source ON leads(source);
CREATE INDEX idx_leads_crm_hash ON leads(crm_record_hash);
