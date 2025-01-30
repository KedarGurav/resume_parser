from resume_parser.llm_integration import initialize_llm, create_chains
from resume_parser.parser import process_resumes
from resume_parser.utils import load_config, validate_directories, setup_logging

def main():
    setup_logging()
    config = load_config()
    validate_directories(config)
    
    llm = initialize_llm(config['google_api_key'])
    chains = create_chains(llm)
    
    process_resumes(
        input_dir=config['input_dir'],
        output_dir=config['output_dir'],
        chains=chains
    )

if __name__ == "__main__":
    main()