from systems.provided.futures_chapter15.estimatedsystem import futures_system


if __name__ == "__main__":

   
    pickle_filepath = 'private.futures_chapter15.estimated_system.pkl'
    

    system = futures_system()
    
    stats = system.accounts.portfolio().stats()
    system.portfolio.get_instrument_correlation_matrix()

    system.cache.pickle(pickle_filepath)

    print('Done')