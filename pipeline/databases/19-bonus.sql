-- Write a SQL script that creates a stored procedure AddBonus that adds
-- a new correction for a student.
DELIMITER //
CREATE PROCEDURE AddBonus(
    IN user_id INT,
    IN project_name VARCHAR(255),
    IN score INT
)
BEGIN
    DECLARE var INT DEFAULT 0;
    DECLARE project_id_value INT DEFAULT 0;

    SELECT COUNT(id) INTO var
    FROM projects 
    WHERE name = project_name;

    IF var = 0 THEN 
        INSERT INTO projects(name)
        VALUES(project_name);
    END IF;

    SELECT id INTO project_id_value
    FROM projects 
    WHERE name = project_name;

    INSERT INTO corrections(user_id, project_id, score)
    VALUES(user_id, project_id_value, score);
END //
DELIMITER ;