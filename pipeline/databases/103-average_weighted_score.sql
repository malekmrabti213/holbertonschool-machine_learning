-- SQL COMMENT
DELIMITER //

DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;

CREATE PROCEDURE ComputeAverageWeightedScoreForUser (user_id INT)
    BEGIN
        SET @avg = (SELECT SUM(corrections.score * projects.weight) / SUM(projects.weight) FROM corrections
            LEFT JOIN projects ON projects.id = corrections.project_id
            WHERE corrections.user_id = user_id);
        UPDATE users SET average_score = @avg WHERE id = user_id;
    END; //

DELIMITER ;