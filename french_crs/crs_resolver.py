import click
import french_crs as fcrs

@click.command()
@click.option('--text',
              help='The input text with potential corefernce chains.')

@click.option('--showchains', 
              default='yes',
              help='Whether to show coreference chains results on screen or not. The default value is "yes"')

@click.option('--visualizer',
              default='enable',
              help='Whether to show coreference chains visualizer results on screen or not. The default value is "enable"')

@click.option('--configfile',
                default='coref-config.json',
              help='The path and the name of the json confing file. The default value is "coref-config.json"')

@click.option('--outputdir', 
                default='.',
              help='The directory that pipeline results will be saved. The default value is "."')

def run_pipelines(text, showchains, visualizer, configfile, outputdir):
    
    if (text==None):

        message="""
Usage: crs-resolver.py [OPTIONS]

    Options:
    --text TEXT        The input text with potential corefernce chains.
    --showchains TEXT  Whether to show coreference chains results on screen or
                        not. The default value is "yes"

    --visualizer TEXT  Whether to show coreference chains visualizer results on
                        screen or not. The default value is "enable"

    --configfile TEXT  The path and the name of the json confing file. The
                        default value is "coref-config.json"

    --outputdir TEXT   The directory that pipeline results will be saved. The
                        default value is "."

    --help             Show this message and exit.

        """
        print(message)
        exit()

    
    print(text)
    print(showchains)
    print(visualizer)
    print(configfile)
    print(outputdir)

if __name__ == '__main__':
    run_pipelines()
