-- Write a script that lists all shows from hbtn_0d_tvshows_rate by their rating.
SELECT t1.title, SUM(t2.rate) AS rating
FROM tv_shows t1
LEFT JOIN tv_show_ratings t2
ON t1.id = t2.show_id
GROUP BY t1.title
ORDER BY SUM(t2.rate) DESC