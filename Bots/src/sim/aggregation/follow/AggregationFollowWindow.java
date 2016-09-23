package sim.aggregation.follow;

import javax.swing.Box;
import javax.swing.JCheckBox;
import javax.swing.JPanel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sim.graphics.Window;
import sim.graphics.World;

@SuppressWarnings("serial")
public class AggregationFollowWindow extends Window{
	int x=600,y=600;
	int pMaxAgentNumber=100,pMinAgentNumber=1,pAgentNumber=10;
	
	JCheckBox cb;
	
	@Override
	public World createWorld() {
		World world = new World(x,y);
		for(int i=0;i<pAgentNumber-1;i++){
			world.addObject(new LineFormationBot(false));
		}
		world.addObject(new LineFormationBot(true));
		return world;
	}

	@Override
	public JPanel getSettings() {
		JPanel panel = new JPanel();
		Box settingsBox = Box.createVerticalBox();
		settingsBox.setAlignmentX(CENTER_ALIGNMENT);
		settingsBox.setAlignmentY(CENTER_ALIGNMENT);
		
		cb = new JCheckBox("Draw Velocity Obstacles",false);
		cb.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				changeDrawVelocityObstacles(e);
			}
		});
		cb.setEnabled(true);
		cb.setSelected(false);
		settingsBox.add(cb);
		panel.add(settingsBox);
		
		return panel;
	}

	public void changeDrawVelocityObstacles(ChangeEvent e){
		if(cb.isSelected()){
			World.drawVelocityObstacle = true;
		}else{
			World.drawVelocityObstacle = false;
		}
	}
}
