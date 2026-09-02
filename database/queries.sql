-- database/queries.sql

-- QUERY 1: Conversion rate by lead source, only considering sources with 200+ leads, sorted by best conversion rate first.
SELECT 
    source,
    COUNT(*) AS total_leads,
    SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) AS converted_leads,
    ROUND(SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS conversion_rate_percent
FROM 
    leads
GROUP BY 
    source
HAVING 
    COUNT(*) >= 200
ORDER BY 
    conversion_rate_percent DESC;


-- QUERY 2: Find duplicate leads.
-- Ideal prevention: 
-- To prevent duplicates from entering the database, we would apply a UNIQUE constraint:
-- ALTER TABLE leads ADD CONSTRAINT uq_crm_record_hash UNIQUE (crm_record_hash);
-- This would reject inserts that share a hash with an existing record.
SELECT 
    crm_record_hash AS duplicate_hash_group,
    COUNT(*) AS duplicate_count
FROM 
    leads
GROUP BY 
    crm_record_hash
HAVING 
    COUNT(*) > 1
ORDER BY 
    duplicate_count DESC;
