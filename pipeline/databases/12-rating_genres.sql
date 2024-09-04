-- Write a script that lists all genres in the database hbtn_0d_tvshows_rate
-- by their rating.
SELECT t4.name, SUM(t2.rate) AS rating
FROM tv_shows t1
LEFT JOIN tv_show_ratings t2
ON t1.id = t2.show_id
LEFT JOIN tv_show_genres t3
ON t1.id = t3.show_id
LEFT JOIN tv_genres t4
ON t3.genre_id = t4.id
WHERE t4.name IS NOT NULL
GROUP BY t4.name
ORDER BY SUM(t2.rate) DESC