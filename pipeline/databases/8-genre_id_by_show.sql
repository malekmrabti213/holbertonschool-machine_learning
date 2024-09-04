-- Write a script that lists all shows contained in hbtn_0d_tvshows that
-- have at least one genre linked.
SELECT t1.title, t2.genre_id
FROM tv_shows t1
INNER JOIN tv_show_genres t2
ON t1.id = t2.show_id
ORDER BY t1.title ASC, t2.genre_id ASC