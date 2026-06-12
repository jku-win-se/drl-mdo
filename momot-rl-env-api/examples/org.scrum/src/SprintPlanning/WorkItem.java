/**
 */
package SprintPlanning;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Work Item</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link SprintPlanning.WorkItem#getIsPlannedFor <em>Is Planned For</em>}</li>
 *   <li>{@link SprintPlanning.WorkItem#getImportance <em>Importance</em>}</li>
 *   <li>{@link SprintPlanning.WorkItem#getEffort <em>Effort</em>}</li>
 *   <li>{@link SprintPlanning.WorkItem#getStakeholder <em>Stakeholder</em>}</li>
 *   <li>{@link SprintPlanning.WorkItem#getName <em>Name</em>}</li>
 * </ul>
 *
 * @see SprintPlanning.SprintPlanningPackage#getWorkItem()
 * @model
 * @generated
 */
public interface WorkItem extends EObject {
   /**
    * Returns the value of the '<em><b>Is Planned For</b></em>' reference list.
    * The list contents are of type {@link SprintPlanning.Sprint}.
    * It is bidirectional and its opposite is '{@link SprintPlanning.Sprint#getCommittedItem <em>Committed Item</em>}'.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Is Planned For</em>' reference list.
    * @see SprintPlanning.SprintPlanningPackage#getWorkItem_IsPlannedFor()
    * @see SprintPlanning.Sprint#getCommittedItem
    * @model opposite="committedItem"
    * @generated
    */
   EList<Sprint> getIsPlannedFor();

   /**
    * Returns the value of the '<em><b>Importance</b></em>' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Importance</em>' attribute.
    * @see #setImportance(int)
    * @see SprintPlanning.SprintPlanningPackage#getWorkItem_Importance()
    * @model required="true"
    * @generated
    */
   int getImportance();

   /**
    * Sets the value of the '{@link SprintPlanning.WorkItem#getImportance <em>Importance</em>}' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @param value the new value of the '<em>Importance</em>' attribute.
    * @see #getImportance()
    * @generated
    */
   void setImportance(int value);

   /**
    * Returns the value of the '<em><b>Effort</b></em>' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Effort</em>' attribute.
    * @see #setEffort(int)
    * @see SprintPlanning.SprintPlanningPackage#getWorkItem_Effort()
    * @model required="true"
    * @generated
    */
   int getEffort();

   /**
    * Sets the value of the '{@link SprintPlanning.WorkItem#getEffort <em>Effort</em>}' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @param value the new value of the '<em>Effort</em>' attribute.
    * @see #getEffort()
    * @generated
    */
   void setEffort(int value);

   /**
    * Returns the value of the '<em><b>Stakeholder</b></em>' reference.
    * It is bidirectional and its opposite is '{@link SprintPlanning.Stakeholder#getWorkitem <em>Workitem</em>}'.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Stakeholder</em>' reference.
    * @see #setStakeholder(Stakeholder)
    * @see SprintPlanning.SprintPlanningPackage#getWorkItem_Stakeholder()
    * @see SprintPlanning.Stakeholder#getWorkitem
    * @model opposite="workitem" required="true"
    * @generated
    */
   Stakeholder getStakeholder();

   /**
    * Sets the value of the '{@link SprintPlanning.WorkItem#getStakeholder <em>Stakeholder</em>}' reference.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @param value the new value of the '<em>Stakeholder</em>' reference.
    * @see #getStakeholder()
    * @generated
    */
   void setStakeholder(Stakeholder value);

   /**
    * Returns the value of the '<em><b>Name</b></em>' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Name</em>' attribute.
    * @see #setName(String)
    * @see SprintPlanning.SprintPlanningPackage#getWorkItem_Name()
    * @model
    * @generated
    */
   String getName();

   /**
    * Sets the value of the '{@link SprintPlanning.WorkItem#getName <em>Name</em>}' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @param value the new value of the '<em>Name</em>' attribute.
    * @see #getName()
    * @generated
    */
   void setName(String value);

} // WorkItem
