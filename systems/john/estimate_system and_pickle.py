from spreadbetting_system import spreadbettingSystem
from syscore.fileutils import get_filename_for_package, get_resolved_pathname

if __name__ == "__main__":
    pickle_filepath = 'private.spreadbettingSystem_100k.estimated_system.pkl'
    config_path = "systems.john.estimation_config.yaml"

    system = spreadbettingSystem(config_path)
    
    stats = system.accounts.portfolio().stats()
    system.portfolio.get_instrument_correlation_matrix()

    system.cache.pickle(pickle_filepath)

    



