import pandas as pd
import sqlite3

con = sqlite3.connect("data/Chinook_Sqlite.sqlite")
cursor = con.cursor()

query = "Select * from artist"
artists_df = pd.DataFrame(cursor.execute(query).fetchall(), columns=['artist_id', 'artist_name'])
# print(artists_df)

query = "Select * from album"
albums_df = pd.DataFrame(cursor.execute(query).fetchall(), columns=['album_id', 'album_title', 'artist_id'])
# print(albums_df)

query = "Select ar.ArtistId, ar.Name, al.Title from album al inner join artist ar on ar.ArtistId = al.ArtistId where ar.Name = \'AC/DC\'"
result = cursor.execute(query).fetchall()
# print(result)

artist_album = albums_df.set_index('artist_id').join(artists_df, on=['artist_id'])
# print(artist_album)

