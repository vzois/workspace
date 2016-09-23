package sim.formation;

import javax.swing.JPanel;
import sim.geometry.FormationShape;
import sim.graphics.Window;
import sim.graphics.World;

@SuppressWarnings("serial")
public class FormationWindow extends Window {
	int x=700;
	
	@Override
	public World createWorld() {
		World world = new World(x,x);

		for(int i=0;i<70;i++){
			world.addObject(new FormationBot(false,FormationShape.CIRCLE,i));
		}
		return world;
	}

	@Override
	public JPanel getSettings() {
		JPanel panel = new JPanel();
		
		return panel;
	}

}
