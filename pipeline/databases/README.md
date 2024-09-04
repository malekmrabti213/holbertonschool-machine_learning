# Databases

> The "Databases" project is a comprehensive exploration of relational and non-relational database concepts, SQL and NoSQL differences, and advanced functionalities in MySQL. It covers various topics, including table creation, query optimization, and the implementation of stored procedures, functions, views, and triggers. Through a series of SQL and Python scripts, participants gain practical insights into managing databases effectively. This project aims to provide a solid understanding of fundamental database principles and their application, catering to individuals seeking in-depth knowledge of database management and SQL/NoSQL systems.

At the end of this project I was able to solve these conceptual questions:

* What’s a relational database
* What’s a none relational database
* What is difference between SQL and NoSQL
* How to create tables with constraints
* How to optimize queries by adding indexes
* What is and how to implement stored procedures and functions in MySQL
* What is and how to implement views in MySQL
* What is and how to implement triggers in MySQL
* What is ACID
* What is a document storage
* What are NoSQL types
* What are benefits of a NoSQL database
* How to query information from a NoSQL database
* How to insert/update/delete information from a NoSQL database
* How to use MongoDB

## Tasks :heavy_check_mark:

| Filename | Task |
| ------ | ------------------------------------------------- | 
| [0-create_database_if_missing.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/0-create_database_if_missing.sql)| A script that creates the database db_0 in your MySQL server. |
| [1-first_table.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/1-first_table.sql)| A script that creates a table called first_table in the current database in your MySQL server. |
| [2-list_values.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/2-list_values.sql)| A script that lists all rows of the table first_table in your MySQL server. |
| [3-insert_value.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/3-insert_value.sql)| A script that inserts a new row in the table first_table in your MySQL server. |
| [4-best_score.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/4-best_score.sql)| A script that lists all records with a score >= 10 in the table second_table in your MySQL server. |
| [5-average.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/5-average.sql)| A script that computes the score average of all records in the table second_table in your MySQL server. |
| [6-avg_temperatures.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/6-avg_temperatures.sql)| A script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending). |
| [7-max_state.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/7-max_state.sql)| A script that displays the max temperature of each state (ordered by State name). |
| [8-genre_id_by_show.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/8-genre_id_by_show.sql)| A script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked. |
| [9-no_genre.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/9-no_genre.sql)| A script that lists all shows contained in hbtn_0d_tvshows without a genre linked. |
| [10-count_shows_by_genre.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/10-count_shows_by_genre.sql)| A script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each. |
| [11-rating_shows.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/11-rating_shows.sql)| A script that lists all shows from hbtn_0d_tvshows_rate by their rating. |
| [12-rating_genres.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/12-rating_genres.sql)| A script that lists all genres in the database hbtn_0d_tvshows_rate by their rating. |
| [13-uniq_users.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/13-uniq_users.sql)| A SQL script that creates a table users. |
| [14-country_users.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/14-country_users.sql)| A SQL script that creates a table users with new requirements. |
| [15-fans.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/15-fans.sql)| A SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans. |
| [16-glam_rock.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/16-glam_rock.sql)| A SQL script that lists all bands with Glam rock as their main style, ranked by their longevity. |
| [17-store.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/17-store.sql)| A SQL script that creates a trigger that decreases the quantity of an item after adding a new order. |
| [18-valid_email.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/18-valid_email.sql)| A SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed. |
| [19-bonus.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/19-bonus.sql)| A SQL script that creates a stored procedure AddBonus that adds a new correction for a student. |
| [20-average_score.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/20-average_score.sql)| A SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student. |
| [21-div.sql](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/21-div.sql)| A SQL script that creates a function SafeDiv that divides the first by the second number or returns 0 if the second number is equal to 0. |
| [22-list_databases](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/22-list_databases)| A script that lists all databases in MongoDB. |
| [23-use_or_create_database](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/23-use_or_create_database)| A script that creates or uses the database my_db. |
| [24-insert](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/24-insert)| A script that inserts a document in the collection school. |
| [25-all](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/25-all)| A script that lists all documents in the collection school. |
| [26-match](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/26-match)| A script that lists all documents with name="Holberton school" in the collection school |
| [27-count](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/27-count)| A script that displays the number of documents in the collection school. |
| [28-update](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/28-update)| A script that adds a new attribute to a document in the collection school. |
| [29-delete](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/29-delete)| A script that deletes all documents with name="Holberton school". |
| [30-all.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/30-all.py)| Lists all documents in a collection. |
| [31-insert_school.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/31-insert_school.py)| Inserts a new document in a collection based on kwargs. |
| [32-update_topics.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/32-update_topics.py)| Changes all topics of a school document based on the name. |
| [33-schools_by_topic.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/33-schools_by_topic.py)| Returns the list of school having a specific topic. |
| [34-log_stats.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/pipeline/databases/34-log_stats.py)| Gets stats about Nginx logs stored in MongoDB. |


### Try It On Your Machine :computer:
```bash
git clone https://github.com/otalorajuand/holbertonschool-machine_learning.git
cd pipeline/databases
```