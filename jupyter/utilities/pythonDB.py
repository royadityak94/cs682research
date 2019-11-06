import mysql.connector
def writeToDB(optimizer, activation, architecture, bag, label=None, keywords=None):
    loss, accuracy, mae, mse, train_loss_dict, train_accuracy_dict, train_mae_dict, train_mse_dict = bag
    db_conn = mysql.connector.connect(host="localhost", user="root", passwd="1234",  database='cs682research')
    db_cursor = db_conn.cursor(buffered=True)
    insert_query = "INSERT INTO research_exploration VALUES('{}', '{}', '{}', \"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\", '{}', '{}');"\
    .format(optimizer, activation, architecture, (train_loss_dict), str(train_accuracy_dict), \
    str(train_mae_dict), str(train_mse_dict), str(loss), str(accuracy), str(mae), str(mse), label, keywords)
    db_cursor.execute(insert_query)
    db_conn.commit()
    db_cursor.close()
    db_conn.close()
    print ("Successfully written to the database...")
    return